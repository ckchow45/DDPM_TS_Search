import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from tqdm import tqdm
import os
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from copy import deepcopy
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import mdtraj as md
from scipy.spatial.transform import Rotation
from collections import defaultdict

# helper functions
def A(x):   return 0.1 * x           # angstroms converted to nm
def deg(x): return np.radians(x)     # degrees converted to rad

def compute_angle(a, b, c):
    """Compute angle (in radians) at point b formed by points a-b-c."""

    # create vectors pointing from a to b and b to c
    ba = a - b # vector pointing from b to a
    bc = c - b # vector pointing from b to a

    # normalize both vectors to unit length to prevent length affecting the angle
    ba /= np.linalg.norm(ba) + 1e-8
    bc /= np.linalg.norm(bc) + 1e-8

    # compute the dot product between ba and bc (cosine of the angle)
    # np.clip used to constrain result to [-1, 1] to avoid floating-point errors outside acos 
    return np.arccos(np.clip(np.dot(ba, bc), -1.0, 1.0)) # return result in radians

def place_atom(a, b, c, r, theta, phi):
    """
    Places a new atom D in 3D space, using three existing atoms A, B, C and known geometry:
        a, b, c - known atom positions as 3D vectors (numpy arrays of shape (3,))
        r - bond length/distance from atom C to new atom D
        theta - bond angle BCD (in radians), (angle between vector CD and CB)
        phi - dihedral angle ABCD (in radians), (rotation about bond BC)

    Function explanation for people who have no background in geometric reconstruction
    Imagine there are 3 atoms placed in 3D space: A, B, and C
        Now place a fourth atom, D, such that:
            The bond length between C and D is a known distance (r): 
            The bond angle BCD is some known value (theta)
            The torsion angle ABCD (dihedral) is some rotation (phi) around the axis BC
        In laymans terms
            Place D r units away from C
            Make sure the angle between atoms B-C-D is theta
            Rotate D around the bond BC by phi degrees
    
    Definitions:
        Orthogonal vectors are perpendicular (90 degree angle) to each other.
        A normal vector is a vector that is perpendicular to a surface or plane. 
        2 vectors in an inner product space are orthonormal if they are orthogonal unit vectors
        To find a vector perpendicular to a plane, use the cross product (eg: nb = AB * BC).       
        Normalization means turning a vector into a unit vector (length = 1), keeping the direction but removing the magnitude.

    Step 1: Define Coordinate Frame
        We want to place atom D relative to known atoms A, B, and C.
        To do this, we construct a local coordinate system around atom C.
        Define:
            bc = C - B: the direction from B to C
            ab = B - A: the direction from A to B
        Construct a perpendicular vector to the plane ABC:
            nb = AB * BC = vector normal to the ABC plane
        Construct a third vector:
            nbc = BC * nb: ensures a right-handed 3D frame
        We now have three orthonormal vectors:
            bc = forward axis
            nbc = "left-right" axis
            nb = "up-down" axis

    Step 2: Use Spherical Coordinates
        place the atom D using internal coordinates:
            Distance from C to D = r
            Angle from BC to CD = theta (bond angle)
            Twist around BC = phi (dihedral)
        Express vector CD in the local coordinate system: CD = -rcos(theta) * bc + rsin(theta) * cos(phi) * nbc + rsin(theta) * sin(phi) * nb
        Essentially:
            Move r units in the direction theta from BC
            Rotate around the BC axis by phi
    
    Step 3: Find global coordinates of D
        Compute vector CD (from C to D), then add it to the position of C
    """
    # construct local reference frame 
    bc = c - b # vector from atom B to C
    bc /= np.linalg.norm(bc) + 1e-8 # normalize bc to get a unit vector. Used as primary axis of the local frame, 1e-8 avoids division by zero

    ab = b - a # vector from A to B
    nb = np.cross(ab, bc) # normal vector to the plane formed by atoms A-B-C (via cross product)
    nb /= np.linalg.norm(nb) + 1e-8 # normalize to get a unit vector orthogonal to plane ABC

    nbc = np.cross(bc, nb) # vector orthogonal to both bc and nb, forming a right-handed local coordinate system
    nbc /= np.linalg.norm(nbc) + 1e-8 # normalize to get unit vector
    # These 3 orthonormal vectors (bc, nb, nbc) now span a local 3D space with C as origin

    # constructs vector from C to the new atom D using: r - (bond length), theta - angle from the -bc axis, phi - rotation around the BC axis
    # (-bc): points away from C along the extension of BC
    # nbc: points in the plane perpendicular to bc, rotated by phi
    # nb: points out of the plane (torsional direction)
    # d_local is the final displacement from C to D in 3D space
    d_local = (r * np.cos(theta)) * (-bc) \
             + (r * np.sin(theta) * np.cos(phi)) * nbc \
             + (r * np.sin(theta) * np.sin(phi)) * nb

    return c + d_local # translate vector back into global coordinates by adding the vector to point C

# deprecated function
def reconstruct_hire_cg_v1(dihedrals, residue_types, bead_orders, bond_lengths, bond_angles):
    """
    Takes a sequence of dihedral angles and reconstructs a coarse-grained 3D molecular structure frame-by-frame
        dihedrals - (n_frames, n_dihedrals), giving backbone and base angle info.
        residue_types - list like ["A", "G", "C", "U", ...], one per nucleotide.
        bead_orders: - maps residue type. An ordered list of bead names (e.g., ["P", "O5", ..., "X1", "X2"]).
        bond_lengths - dictionary of average bond lengths (in nm) between bead pairs.
        bond_angles - dictionary of average angles (in radians) between bead triplets.
    """
    # convert to numpy array
    if isinstance(dihedrals, torch.Tensor): # if dihedrals are in tensor then conver to numpy array
        dihedrals = dihedrals.detach().cpu().numpy()
    else:
        dihedrals = np.asarray(dihedrals)

    # if input is a batch of one sample (1, T, D), flatten to (T, D).
    if dihedrals.ndim == 3 and dihedrals.shape[0] == 1:
        dihedrals = dihedrals[0]  # (T, D)

    # error catching to make sure inputs are in 2D 
    if dihedrals.ndim != 2:
        raise ValueError(f"Dihedrals must be 2D (frames * angles); got shape {dihedrals.shape}")

    # extract relevant information for calculations
    n_frames = dihedrals.shape[0] # number of timesteps/frames in the path.
    total_beads = sum(len(bead_orders[r]) for r in residue_types) # total number of beads across all residues
    coords = np.zeros((n_frames, total_beads, 3)) # array to store XYZ coordinates of all beads.

    # offsets to keep track of where we are in the full bead/dihedral arrays as we move through residues
    # offsets are incremented per residue
    bead_ofs = 0 # index into the output coordinate array, tracking the running total of beads already placed (EG: A has 7 beads so will increment by 7 to get next residue slice)
    dih_ofs = 0 # index into the flat dihedral array, tracking the running total of dihedrals already read (EG: A has 10 dihedrals so will increment by 10 to get next residue slice)

    for r_idx, rtype in enumerate(residue_types): # loop over the number of residues and extract the residue name
        beads = bead_orders[rtype] # get list of beads for the current residue
        n_beads = len(beads) # get number of beads in residue
        n_dihs = 10 if rtype in ("A", "G") else 8 # decide number of dihedrals to read (10 for purines, 8 for pyrimidines)

        for f in range(n_frames): # loop over each frame
            # setup short lambda functions
            def BL(i, j):
                key = (rtype, beads[i], beads[j])
                if key not in bond_lengths:
                    raise KeyError(f"Missing bond length for key: {key}")
                return float(bond_lengths[key]) # function to get bond length between two bead indices (i ,j) from the given dictionary by giving residue and bead indices, or 0.15 nm default
            def BA(i, j, k):
                print(f"Residue {rtype}, beads: {beads}")
                key = (rtype, beads[i], beads[j], beads[k])
                if key not in bond_angles:
                    raise KeyError(f"Missing bond angle for key: {key}")
                return float(bond_angles[key]) # function to get bond angle from dictionary by giving residue and bead indices, or tetrahedral default.
            def DIH(idx):
                return float(dihedrals[f, dih_ofs + idx].item()) # function to extract the idx-th dihedral angle (from the current residue) at frame f
            
            """
            f - current frame in the trajectory.
            dih_ofs - offset into the flattened dihedrals array.
            dih_ofs + idx - index of the desired dihedral for this residue.
            """

            # Places beads P, O5, C5 in the XY plane. 
            # By anchoring the first three beads in a fixed 2D plane, stable coordinate system from which all subsequent beads can be placed using internal geometry (with bond length, bond angle, dihedral angle)
            # Placement of arbitrary anchor point (P) allows to define the rest of the geometry relative to it
            coords[f, bead_ofs+0] = np.array([0.0, 0.0, 0.0])  # places the first atom (P) at the origin (0, 0, 0) of the 3D space
            coords[f, bead_ofs+1] = np.array([BL(0,1), 0.0, 0.0])  # places the second atom (O5) along the positive x-axis, at a distance equal to the bond length between P and O5. BL(0,1) retrieves that bond length from the constant dictionary.

            # place C5, which is bonded to O5 and forms a known angle with the previous two atoms (P-O5-C5)
            # r = O5-C5 bond length
            # theta = angle at O5 between P and C5 (i.e., angle PO5C5)
            # x = current O5 x-position + r * cos(theta)
            x = coords[f, bead_ofs+1,0] + BL(1,2) * np.cos(BA(0,1,2)) # BL(1,2) = bond length between O5 and C5, BA(0,1,2) = bond angle P-O5-C5, in radians
            y = BL(1,2) * np.sin(BA(0,1,2)) # y = r * sin(theta)
            coords[f, bead_ofs+2] = np.array([x, y, 0.0])  # puts C5 at a point rotated angle theta counterclockwise from the x-axis (O5-C5 bond), with P at the origin and O5 on the x-axis

            if r_idx == 0:
                coords[f, bead_ofs+0] = np.array([0.0, 0.0, 0.0])
                coords[f, bead_ofs+1] = np.array([BL(0,1), 0.0, 0.0])
                x = coords[f, bead_ofs+1,0] + BL(1,2) * np.cos(BA(0,1,2))
                y = BL(1,2) * np.sin(BA(0,1,2))
                coords[f, bead_ofs+2] = np.array([x, y, 0.0])
            else:
                prev_idx = bead_ofs - n_beads
                a = coords[f, prev_idx+1]
                b = coords[f, prev_idx+2]
                c = coords[f, prev_idx+4]
                r = BL(0,1)
                theta = BA(2,4,0)
                phi = DIH(6 if rtype in ("A", "G") else 5)
                coords[f, bead_ofs+0] = place_atom(a, b, c, r, theta, phi)

                coords[f, bead_ofs+1] = place_atom(b, c, coords[f, bead_ofs+0], BL(0,1), BA(4,0,1), DIH(7 if rtype in ("A", "G") else 6))
                coords[f, bead_ofs+2] = place_atom(c, coords[f, bead_ofs+0], coords[f, bead_ofs+1], BL(1,2), BA(0,1,2), DIH(0))

            coords[f, bead_ofs+3] = place_atom(
                coords[f, bead_ofs+0], coords[f, bead_ofs+1], coords[f, bead_ofs+2],
                BL(2,3), BA(1,2,3), DIH(8 if rtype in ("A","G") else 6))

            coords[f, bead_ofs+4] = place_atom(
                coords[f, bead_ofs+1], coords[f, bead_ofs+2], coords[f, bead_ofs+3],
                BL(3,4), BA(2,3,4), DIH(7 if rtype in ("A","G") else 5))

            coords[f, bead_ofs+5] = place_atom(
                coords[f, bead_ofs+2], coords[f, bead_ofs+3], coords[f, bead_ofs+4],
                BL(4,5), BA(3,4,5), DIH(2 if rtype in ("A","G") else 0))

            if rtype in ("A", "G") and n_beads > 6:
                coords[f, bead_ofs+6] = place_atom(
                    coords[f, bead_ofs+3], coords[f, bead_ofs+4], coords[f, bead_ofs+5],
                    BL(5,6), BA(4,5,6), DIH(0))
                
        # update offsets to move to the next residue slice in the flat coordinate array
        bead_ofs += n_beads
        dih_ofs  += n_dihs

    return coords # returns (n_frames, total_beads, 3 coordinates): full 3D CG trajectory.

# currently used function
def reconstruct_hire_cg(dihedrals, residue_types, bead_orders, bond_lengths, bond_angles, used_bond_lengths=None, used_bond_angles=None):
    """
    Takes a sequence of dihedral angles and reconstructs a coarse-grained 3D molecular structure frame-by-frame
        dihedrals - (n_frames, n_dihedrals), giving backbone and base angle info.
        residue_types - list like ["A", "G", "C", "U", ...], one per nucleotide.
        bead_orders: - maps residue type. An ordered list of bead names (e.g., ["P", "O5", ..., "X1", "X2"]).
        bond_lengths - dictionary of average bond lengths (in nm) between bead pairs.
        bond_angles - dictionary of average angles (in radians) between bead triplets.
        used_bond_lengths / angles: optional sets for tracking which entries were used (for diagnostics and debugging)
    """
    # initialize tracking sets if they not provided
    if used_bond_lengths is None:
        used_bond_lengths = set()
    if used_bond_angles is None:
        used_bond_angles = set()

    # convert to numpy array
    if isinstance(dihedrals, torch.Tensor): # if dihedrals are in tensor then convert to numpy array
        dihedrals = dihedrals.detach().cpu().numpy()
    else:
        dihedrals = np.asarray(dihedrals) 

    # if input is a batch of one sample (1, T, D), convert to (T, D).
    if dihedrals.ndim == 3 and dihedrals.shape[0] == 1:
        dihedrals = dihedrals[0]

    # error catching to make sure inputs are in 2D 
    if dihedrals.ndim != 2:
        raise ValueError(f"Dihedrals must be 2D (frames * angles); got shape {dihedrals.shape}")

    # extract relevant information for calculations
    n_frames = dihedrals.shape[0] # number of timesteps/frames in the path.
    total_beads = sum(len(bead_orders[r]) for r in residue_types) # total number of beads across all residues
    coords = np.zeros((n_frames, total_beads, 3)) # array to store XYZ coordinates of all beads

    # offsets to keep track of where we are in the full bead/dihedral arrays as we move through residues
    # offsets are incremented per residue
    bead_ofs = 0 # index into the output coordinate array, tracking the running total of beads already placed (EG: A has 7 beads so will increment by 7 to get next residue slice)
    dih_ofs = 0 # index into the flat dihedral array, tracking the running total of dihedrals already read (EG: A has 10 dihedrals so will increment by 10 to get next residue slice)

    for r_idx, rtype in enumerate(residue_types): # loop over the number of residues and extract the residue name
        beads = bead_orders[rtype] # get list of beads for the current residue
        n_beads = len(beads) # get number of beads in residue
        n_dihs = 10 if rtype in ("A", "G") else 8 # decide number of dihedrals to read (10 for purines, 8 for pyrimidines)

        for f in range(n_frames): # loop over each frame
            # setup short lambda functions, used for bond length, bond angle, and dihedral lookup in reconstruction
            def BL(i, j):
                """
                Function BL (Bond Length) used to retrieve the bond length between bead i and bead j for the current residue.
                """
                key = (rtype, beads[i], beads[j]) # lookup key based on rtype (the current residue type (e.g., "A", "G")), beads[i], beads[j] (names of the two beads involved (e.g., "C4", "P"))
                used_bond_lengths.add(key) # records the usage of this bond length key in a set for later diagnostics (to track which constants were actually used
                if key not in bond_lengths: # checks if the bond length value exists in the bond_lengths dictionary
                    raise KeyError(f"Missing bond length for key: {key}") # raise error if bond length does no exist in provided dictionary
                return float(bond_lengths[key]) # returns the bond length (converted to a float for consistency)

            def BA(i, j, k):
                """
                Function BA (Bond Angle) to retrieve the angle formed by three beads: i-j-k
                """
                key = (rtype, beads[i], beads[j], beads[k]) # key for the bond angle dictionary using residue type, and 3 provided beads (i,j,k)
                used_bond_angles.add(key) # records the usage of this bond length key in a set for later diagnostics (to track which constants were actually used
                if key not in bond_angles: # checks if the bond angle value exists in the bond_lengths dictionary
                    raise KeyError(f"Missing bond angle for key: {key}") # raise error if bond length does no exist in provided dictionary
                return float(bond_angles[key]) # returns the bond angle in radians(converted to a float for consistency)

            def DIH(idx): 
                """
                Function to retrieve the dihedral angle at a specific index for the current frame f
                    The indices used to lookup dihedrals are determined by the predefined purine and pyrimidine quadruplets so DIH(0) corresponds to the first dihedral in the list (e.g for purines: R4-R1-X1-X2)
                """
                return float(dihedrals[f, dih_ofs + idx].item()) # retrieve the dihedral angle at offset dih_ofs + idx for frame f

            # place first 3 beads for this current frame f
            # by anchoring the first three beads, all subsequent beads can be placed using internal geometry (with bond length, bond angle, dihedral angle)
            # placement of arbitrary anchor point (P) allows to define the rest of the geometry relative to it
            coords[f, bead_ofs+0] = np.array([0.0, 0.0, 0.0]) # P at origin
            coords[f, bead_ofs+1] = np.array([BL(0,1), 0.0, 0.0]) # place O5 at distance equal to the bond length between P and O5', retrieved via BL(0,1)
            # computes the x and y position of C5' with
                # bond length between O5'-C5' with BL(1,2)
                # angle of P-O5'-C5' using BA(0,1,2)
            x = coords[f, bead_ofs+1,0] + BL(1,2) * np.cos(BA(0,1,2)) 
            y = BL(1,2) * np.sin(BA(0,1,2)) 
            coords[f, bead_ofs+2] = np.array([x, y, 0.0]) # place C5 in calculated (x, y, 0) position

            if r_idx == 0:
                pass # for the first residue, the first 3 beads were just placed manually so skip further placement
            else: # For all other residues, find the index of the previous residue's beads
                # identify reference atoms from the previous residue: O5', C5', and C4' to position the P bead of the current residue
                prev_idx = bead_ofs - n_beads # get index of previous residue
                a = coords[f, prev_idx+1]  # O5'
                b = coords[f, prev_idx+2]  # C5'
                c = coords[f, prev_idx+3]  # C4'
                r = BL(0,1) # get bond length P-O5'
                theta = BA(1,2,3)  # get angle O5-C5-C4
                phi = DIH(6 if rtype in ("A", "G") else 5) # get dihedral to place P, dihedral index 6 is used for purines (A/G), 5 for pyrimidines (U/C) due to different lengths in expected dihedral lists in purine and pyrimidines
                coords[f, bead_ofs+0] = place_atom(a, b, c, r, theta, phi) # P bead placement with place_atom function using O5', C5', and C4' beads from previous residue as reference

                coords[f, bead_ofs+1] = place_atom(b, c, coords[f, bead_ofs+0], BL(0,1), BA(2,3,0), DIH(7 if rtype in ("A", "G") else 6)) # place O5' of the current residue using C5', C4', and newly placed P bead
                coords[f, bead_ofs+2] = place_atom(c, coords[f, bead_ofs+0], coords[f, bead_ofs+1], BL(1,2), BA(0,1,2), DIH(0)) # place C5' using the 3 most recently placed beads: C4', P, and O5'

            # place remaining beads (C4, C1, X1, X2)
            coords[f, bead_ofs+3] = place_atom( # place C4' using P, O5', and C5' as reference
                coords[f, bead_ofs+0], coords[f, bead_ofs+1], coords[f, bead_ofs+2],
                BL(2,3), BA(1,2,3), DIH(8 if rtype in ("A","G") else 6)) # uses dihedral index 8 for purines, 6 for pyrimidines

            coords[f, bead_ofs+4] = place_atom( # place C1' using O5', C5', and C4' as reference
                coords[f, bead_ofs+1], coords[f, bead_ofs+2], coords[f, bead_ofs+3],
                BL(3,4), BA(2,3,4), DIH(7 if rtype in ("A","G") else 5)) # uses dihedral index 7 (purines) or 5 (pyrimidines)

            coords[f, bead_ofs+5] = place_atom( # place X1 base bead using C5', C4', and C1' as reference
                coords[f, bead_ofs+2], coords[f, bead_ofs+3], coords[f, bead_ofs+4],
                BL(4,5), BA(3,4,5), DIH(2 if rtype in ("A","G") else 0)) # uses dihedral index 2 (purines) or 0 (pyrimidines)

            if rtype in ("A", "G") and n_beads > 6: # check if the residue is a purine (A or G) and has more than 6 beads as purines have X2
                coords[f, bead_ofs+6] = place_atom( # place X2 using C4', C1', and X1 as references
                    coords[f, bead_ofs+3], coords[f, bead_ofs+4], coords[f, bead_ofs+5],
                    BL(5,6), BA(4,5,6), DIH(1)) # using dihedral index 1.

        # move the bead and dihedral index forward to prepare for placing the next residue
        bead_ofs += n_beads
        dih_ofs  += n_dihs

    return coords # returns the full 3D coordinate array (shape(n_frames, total_beads, 3 coordinates))

def normalize_dihedrals(dihedrals):
    """
    Normalize dihedral angles from [-180 degrees, +180 degrees] to [-1, 1]
    Args:
        dihedrals: Tensor or array of shape (..., n_dihedrals) in radians
    Returns:
        Normalized dihedrals in [-1, 1]
    """
    return dihedrals / np.pi

def compute_dihedrals(coords, quadruplets):
    """
    Compute dihedral angles from bead coordinates.
        - coords: A numpy array of shape (n_frames, n_beads, 3) containing 3D coordinates of beads across multiple frames. 
        - quadruplets: A list of tuples (i, j, k, l). Each tuple (i, j, k, l) specifies four bead indices that define one dihedral

    Function explanation for people who have no background in geometric reconstruction (me)
    A dihedral angle is the angle between two planes, defined by four points (atoms/beads). 
        EG: (p1-p2-p3-p4)
        Plane 1: formed by points (p1, p2, p3)
        Plane 2: formed by points (p2, p3, p4)
    The angle is how much plane 2 is twisted relative to plane 1, measured around the central bond
    It's positive or negative depending on clockwise/counter-clockwise twist and zero when planes are aligned
    
    Definitions:
        A normal vector is a vector that is perpendicular to a surface or plane. 
        To find a vector perpendicular to a plane, use the cross product (eg: nb = AB * BC).       
        Normalization means turning a vector into a unit vector (length = 1), keeping the direction but removing the magnitude.

    Step 1: Extract bond vectors
        There are 3 bonds connecting the 4 points represented as vectors
    
    Step 2: Compute normals to the planes
        Find vectors perpendicular to the two planes
        n1 sticks out of plane 1 
        n2 sticks out of plane 2

    Step 3: Normalize
        We want to work with directions, not magnitudes to prevent magnitude from affecting the angle
        Convert n1, n2, and b2 into unit vectors
    
    Step 4:
        Find the angle between the two normals using atan2
    """
    angles = [] # list to store computed dihedrals
    for (i, j, k, l) in quadruplets: # iterate over each set of 4 bead indices (i, j, k, l) in the predefined quadruplets
        # extract the coordinates of the 4 beads in all frames
        p1 = coords[:, i]
        p2 = coords[:, j]
        p3 = coords[:, k]
        p4 = coords[:, l]

        # Calculates the vectors connecting consecutive beads (the bond vectors)
        b1 = p2 - p1 # vector from bead i to j.
        b2 = p3 - p2 # vector from bead j to k.
        b3 = p4 - p3 # vector from bead k to l.

        # cross product to find normals to the two planes
        n1 = np.cross(b1, b2) # normal to the plane formed by (p1, p2, p3). n1 is perpendicular to the first plane.
        n2 = np.cross(b2, b3) # normal to the plane formed by (p2, p3, p4). n2 is perpendicular to the second plane.

        # normalize n1, n2, and b2 to unit vectors to avoid scaling artifacts and remove lengths, 1e-8 is added to prevent division-by-zero and floating-point instability when normalizing near-zero vectors
        n1_norm = np.linalg.norm(n1, axis=1, keepdims=True) + 1e-8 
        n2_norm = np.linalg.norm(n2, axis=1, keepdims=True) + 1e-8
        b2_norm = np.linalg.norm(b2, axis=1, keepdims=True) + 1e-8

        n1 /= n1_norm
        n2 /= n2_norm
        b2 /= b2_norm

        # Compute dihedral using atan2
        x = np.sum(n1 * n2, axis=1) # dot product of n1 and n2 (cosine of the angle between normals or, x is the cosine of the dihedral
        y = np.sum(np.cross(n1, n2) * b2, axis=1) # Projection of the cross product (n1 * n2) onto b2 (sine of the angle).
        angle = np.arctan2(y, x) # Uses arctan2(y, x) to compute the dihedral angle in the range [-π, π].
        angles.append(angle) # append to storage list
        
    return np.stack(angles, axis=1) # convert the list of angles (one per quadruplet) into a 2D numpy array of shape (n_frames, n_quadruplets).


def periodic_l2_loss(pred, target):
    """
    Compute L2 loss that respects periodicity of angles in [-1, 1] scaled space
    Assumes angles are normalized (divided by pi)

    pred - a tensor of predicted angle values in [-1, 1]
    target - a tensor of target (true) angle values in [-1, 1]
    """
    diff = pred - target # raw/naive different in values
    wrapped_diff = torch.remainder(diff + 1.0, 2.0) - 1.0  # mapping any difference back into the range [-1, 1]
    # diff + 1.0: shifts range from [-2, 2] to [-1, 3]
    # torch.remainder(..., 2.0): wraps values to [0, 2)
    # ... - 1.0: shifts back to [-1, 1)
    return torch.mean(wrapped_diff ** 2) # compute mean squared error but applied to the wrapped differences

# function for saving the inference output into a XYZ file
def save_xyz(generated_path, filename, n_atoms, atom_names=None):
    """
    Save a trajectory as an XYZ file.
    
    Parameters:
    - coordinates: multi-dimensional (frames, atoms, 3 coordinates) array of frames
    - filename: output filename
    - atom_names: list of atom names (optional, defaults to 'C')
    """
    path = generated_path.squeeze(0)  # Shape: (50, F), remove the batch number
    xyz = path.reshape(-1, n_atoms, 3) # turn into format (number of frames, number of atoms, XYZ coordinates for each atom)
    coordinates = xyz.cpu().numpy() # convert to numpy array

    frames, atoms, _ = coordinates.shape
    if atom_names is None: # if no atom symbols are given
        atom_names = ['C'] * atoms  # default to carbon for all atoms

    with open(filename, 'w') as file:
        for f in range(frames): # loop over every frame
            file.write(str(atoms) + "\n") # number of atoms
            file.write(f"Frame {f+1}\n") # comment line just saying which frame the following information is, +1 to avoid 0 indexing
            for i in range(atoms): # loop over every atom
                x, y, z = coordinates[f, i] # extract coordinates from each frame and atom
                file.write(f"{atom_names[i]} {x:.5f} {y:.5f} {z:.5f}\n") # write each atom XYZ coordinate, the .5f is python string formatting to only show 5 decimal places

# helper function to identify purines.
def is_purine(nt): 
    return nt in ['A', 'G']

# deprecated function, can be run but introduces ring artifacts into the molecular structure due to issues with P bead placement
def geometry_parameters_v1(trajectory_folder: str, prmtop_file: str, bead_orders, bond_length_constants, bond_angles_constants, atomic_weights):
    """
    Returns (bead_order, bond_lengths[nm], bond_angles[rad]).

    Backbone geometry is taken from the hard-coded constant tables given by the user
    (bond_length_constants & bond_angles_constants).

    Base beads (X1, X2) do not have fixed geometry, so we scan
    trajectories to measure (C1-X1) and (X1-X2) statistics.
    """
    # lists to store values 
    bond_len   = defaultdict(list)
    bond_angle = defaultdict(list)

    # extract the bond lengths from constant tables
    for (res,b1,b2), val in bond_length_constants.items():
        if res=="*":
            for r in bead_orders:
                bond_len[(r,b1,b2)] = A(val) # helper function A() called to convert from angstroms to nm
        else:
            bond_len[(res,b1,b2)] = A(val)

    # extract the bond angles from constant tables
    for (res,b1,b2,b3), val in bond_angles_constants.items():
        if res=="*":
            for r in bead_orders:
                bond_angle[(r,b1,b2,b3)] = deg(val) # helper function deg() called to convert degrees to radians
        else:
            bond_angle[(res,b1,b2,b3)] = deg(val)

    # scan trajectories again to calculate COM geometry 
    CG_BASES = {
    "A1": ["N7", "N9", "C4", "C5", "C8"],
    "A2": ["N1", "C2", "N3", "C4", "C5", "N6", "C6"],
    "C1": ["N1", "C2", "N3", "N4", "C4", "C5", "C6", "O2"],
    "G1": ["N7", "N9", "C4", "C5", "C8"],
    "G2": ["N1", "N2", "C2", "N3", "C4", "C5", "C6", "O6"],
    "T1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"],
    "U1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"]
    }  

    # loop over every path file
    for fname in tqdm(os.listdir(trajectory_folder), desc="Base COM harvesting"):
        if not fname.endswith(".mdcrd"): 
            continue # ignore files that do not end in mdcrd

        # load in the paths using MDTraj
        traj = md.load_mdcrd(os.path.join(trajectory_folder,fname), top=prmtop_file)
        top = traj.topology
        nF  = traj.n_frames

        # path file processing 
        for res in top.residues: # loop over each residue
            rt = res.name.strip()[0] # get residue name
            if rt not in bead_orders: 
                continue # if the residue symbol is not in the bead_orders provided then continue 
            amap = {a.name:a.index for a in res.atoms} # create a dictionary of atom names to atom index in the residue

            # extract coords for C1 to compute the bond length to X1
            C1 = traj.xyz[:, amap["C1'"], :]
            C4 = traj.xyz[:, amap["C4'"], :]

            # build COM beads (X1/X2) for this residue
            base_coms = [] # list to store X1/X2
            tags = [f"{rt}1", f"{rt}2"] if rt in ("A","G") else [f"{rt}1"] # decide 1 or 2 base beads to process. For purines (A, G): [A1, "A2"] or [G1, G2], pyrimidines (C, U, T): [C1], etc.
            for tag in tags: # for 1 or 2 beads
                clist, elist = [], [] # coordinate and element lists for storage: clist = list of arrays, each (n_frames, 3 coordinates) for each atom. elist = atom elements, aligned with each coordinate list in coords_list
                for at in CG_BASES[tag]: # For each atom in the CG base group 
                    if at in amap: # check if the atom exists in the residue atom map
                        clist.append(traj.xyz[:, amap[at], :]) # extract atom XYZ coordinates and append to list
                        elist.append(top.atom(amap[at]).element.symbol) # extract atom element and append to list
                coords = np.stack(clist, axis=1) # aligns/stacks all selected atoms side-by-side for a given frame so we can treat each frame independently
                com = np.array([compute_com(coords[i], elist, atomic_weights) for i in range(nF)]) # for each frame, compute the center of mass of the atoms
                base_coms.append((tag, com)) # append centre of mass to list

            # now accumulate bond length / angle for C1-X1 and X1-X2 etc.
            if base_coms: # if at least one bead calculated, extract and treat as X1
                tag1, X1 = base_coms[0] # tag1 is a string like "A1" or "C1", depending on the residue, X1 is a NumPy array of shape (frames, 3) containing the COM coordinates for each frame
                for i in range(nF): # For every frame
                    dist = np.linalg.norm(X1[i]-C1[i]) # compute the Euclidean distance between C1 and X1
                    key1 = (rt, "C1", tag1)
                    if not isinstance(bond_len[key1], list):
                        bond_len[key1] = [bond_len[key1]]
                    bond_len[key1].append(dist) # append it in a dictionary bond_len under a key like (residue type ='A', bead pair = C1-A1).

                    angle = compute_angle(C4[i], C1[i], X1[i]) # compute the bond angle C4'-C1'-X1 using compute_angle (computes the angle at b between vectors a-b and c-b)
                    key_angle = (rt, "C4", "C1", tag1)
                    if not isinstance(bond_angle[key_angle], list):
                        bond_angle[key_angle] = [bond_angle[key_angle]]
                    bond_angle[key_angle].append(angle) # store angle under key like ("A", "C4", "C1", "A1")

                if len(base_coms)==2: # if a second base bead exists
                    tag2, X2 = base_coms[1] # treat as X2 and do same procedure
                    for i in range(nF):
                        dist = np.linalg.norm(X2[i]-X1[i])
                        key1 = (rt, tag1, tag2)
                        if not isinstance(bond_len[key1], list):
                            bond_len[key1] = [bond_len[key1]]
                        bond_len[key1].append(dist)

                        angle = compute_angle(C1[i], X1[i], X2[i])
                        key_angle = (rt,"C1", tag1, tag2)
                        if not isinstance(bond_angle[key_angle], list):
                            bond_angle[key_angle] = [bond_angle[key_angle]]
                        bond_angle[key_angle].append(angle)

    # finalise averages for base bonds and bond lengths
    for k,v in bond_len.items(): # Loop through every entry in the bond_len dictionary.
        # k is a tuple like ("A", "C1", "A1"), representing a bond between two beads in a particular residue type.
        # v is either: A single float value (already processed constant from bond table) or a list of distances, Any entry that came from dynamic measurement (not hardcoded) is a list of distances.
        if isinstance(v,list): # distinguishes between the two v cases, float or list
            bond_len[k] = float(np.mean(v)) # for each bond key whose value is a list of distances: Compute the mean bond length across all frames, convert the result to a float, replace the list with this averaged value.
            
    # same procedure as bond lengths but for bond angles
    for k, v in bond_angle.items():
        if isinstance(v, list):
            bond_angle[k] = float(np.mean(v))

    return bead_orders, bond_len, bond_angle

# previous geometry parameters function placed all Ps in the same position, causing all residues to be in the same spot, this created the ring artifact
# also deprecated, this version accumulates bond length and angle information for base beads, was used to compare with predefined bond lengths and angles 
def geometry_parameters_accumulate(trajectory_folder: str, prmtop_file: str, bead_orders, bond_length_constants, bond_angles_constants, atomic_weights):
    """
    Returns (bead_order, bond_lengths[nm], bond_angles[rad]).

    Backbone geometry is taken from the hard-coded constant tables given by the user
    (bond_length_constants & bond_angles_constants).

    Base beads (X1, X2) do not have fixed geometry, so we scan
    trajectories to measure (C1-X1), (X1-X2), and additional angles including
    cross-residue angles.
    """

    # Helper functions to convert units
    A = lambda x: x * 0.1  # convert Å to nm
    deg = lambda x: np.radians(x)

    # lists to store values 
    bond_len   = defaultdict(list)
    bond_angle = defaultdict(list)

    # extract the bond lengths from constant tables
    for (res,b1,b2), val in bond_length_constants.items():
        if res=="*":
            for r in bead_orders:
                bond_len[(r,b1,b2)] = A(val)
        else:
            bond_len[(res,b1,b2)] = A(val)

    # extract the bond angles from constant tables
    for (res,b1,b2,b3), val in bond_angles_constants.items():
        if res=="*":
            for r in bead_orders:
                bond_angle[(r,b1,b2,b3)] = deg(val)
        else:
            bond_angle[(res,b1,b2,b3)] = deg(val)
    # scan trajectories again to calculate COM geometry 
    CG_BASES = {
        "A1": ["N7", "N9", "C4", "C5", "C8"],
        "A2": ["N1", "C2", "N3", "C4", "C5", "N6", "C6"],
        "C1": ["N1", "C2", "N3", "N4", "C4", "C5", "C6", "O2"],
        "G1": ["N7", "N9", "C4", "C5", "C8"],
        "G2": ["N1", "N2", "C2", "N3", "C4", "C5", "C6", "O6"],
        "T1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"],
        "U1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"]
    }

    for fname in tqdm(os.listdir(trajectory_folder), desc="Base COM harvesting"): # loop over every path file
        if not fname.endswith(".mdcrd"): continue # ignore files that do not end in mdcrd

        # load in the paths using MDTraj
        traj = md.load_mdcrd(os.path.join(trajectory_folder, fname), top=prmtop_file)
        top = traj.topology # extract path topology
        nF  = traj.n_frames # extract number of frames
        residues = list(top.residues) # extract residues

        # path file processing 
        for i, res in enumerate(residues): # loop over each residue
            rt = res.name.strip()[0] # extract the residue names first character (e.g., "A", "G", etc.) to use as residue type
            if rt not in bead_orders: continue # if the residue symbol is not in the bead_orders provided then skip
            amap = {a.name:a.index for a in res.atoms} # create a dictionary of atom names to atom index in the residue

            # extract positions of bead sugar atoms across all frames
            C1 = traj.xyz[:, amap["C1'"], :]
            C4 = traj.xyz[:, amap["C4'"], :]
            O5 = traj.xyz[:, amap["O5'"], :]
            C5 = traj.xyz[:, amap["C5'"], :]

            # define base bead tags: purines have two base beads (e.g. A1, A2); pyrimidines only one.
            base_coms = []
            tags = [f"{rt}1", f"{rt}2"] if rt in ("A","G") else [f"{rt}1"]

            # collecting atom coordinates and elements for each base bead
            for tag in tags: # 1 or 2 depending on if purine or pyrimidine
                clist, elist = [], [] # coordinate and element lists for storage: clist = list of arrays, each (n_frames, 3 coordinates) for each atom. elist = atom elements, aligned with each coordinate list in coords_list
                for at in CG_BASES[tag]: # for each atom in the CG base group 
                    if at in amap: # check if the atom exists in the residue atom map
                        clist.append(traj.xyz[:, amap[at], :]) # extract atom XYZ coordinates and append to list
                        elist.append(top.atom(amap[at]).element.symbol) # extract atom element and append to list
                if clist: # if any atoms were found,
                    coords = np.stack(clist, axis=1)
                    com = np.array([np.average(coords[j], axis=0, weights=[atomic_weights[e] for e in elist]) for j in range(nF)]) # compute the mass-weighted center of mass for each frame
                    base_coms.append((tag, com)) # append centre of mass to list

            # now accumulate bond length / angle for C1-X1 and X1-X2 etc.
            if base_coms: # use the first base bead (X1), always exists if base_coms is non-empty.
                tag1, X1 = base_coms[0] # tag1 is a string like "A1" or "C1", depending on the residue, X1 is an array of shape (frames, 3) containing the X1 coordinates for each frame
                for j in range(nF): # for every frame
                    dist = np.linalg.norm(X1[j]-C1[j]) # compute the Euclidean distance between C1 and X1
                    if not isinstance(bond_len[(rt, "C1", tag1)], list): # if this is the first time seeing this key, initialize it as a list
                        bond_len[(rt, "C1", tag1)] = [bond_len[(rt, "C1", tag1)]] # store the bond length in the dictionary under key (residue_type, "C1", "X1")
                    bond_len[(rt, "C1", tag1)].append(dist) # store this bond length into the dictionary, appending it to a list 

                    angle1 = compute_angle(C4[j], C1[j], X1[j]) # compute the angle C4'-C1'-X1 for this frame
                    if not isinstance(bond_angle[(rt, "C4", "C1", tag1)], list): # if this is the first time seeing this key, initialize it as a list
                        bond_angle[(rt, "C4", "C1", tag1)] = [bond_angle[(rt, "C4", "C1", tag1)]] # store the bond angle in the dictionary under key (residue_type, "C4", "C1", "X1")
                    bond_angle[(rt, "C4", "C1", tag1)].append(angle1) # append the angle to the corresponding entry in the angle dictionary

                    angle2 = compute_angle(O5[j], C4[j], X1[j]) # compute another angle: O5'-C4'-X1
                    if not isinstance(bond_angle[(rt, "O5", "C4", tag1)], list):
                        bond_angle[(rt, "O5", "C4", tag1)] = [bond_angle[(rt, "O5", "C4", tag1)]]
                    bond_angle[(rt, "O5", "C4", tag1)].append(angle2) # append this angle

                # if purine (X2 exists), measure X1-X2 bond and angle
                if len(base_coms) == 2:
                    tag2, X2 = base_coms[1]
                    for j in range(nF):
                        # measure and store X1-X2 bond length
                        dist = np.linalg.norm(X2[j] - X1[j])
                        if not isinstance(bond_len[(rt, tag1, tag2)], list):
                            bond_len[(rt, tag1, tag2)] = [bond_len[(rt, tag1, tag2)]]
                        bond_len[(rt, tag1, tag2)].append(dist)

                        # measure and store angle C1'-X1-X2
                        angle = compute_angle(C1[j], X1[j], X2[j])
                        if not isinstance(bond_angle[(rt, "C1", tag1, tag2)], list):
                            bond_angle[(rt, "C1", tag1, tag2)] = [bond_angle[(rt, "C1", tag1, tag2)]]
                        bond_angle[(rt, "C1", tag1, tag2)].append(angle)

            # Cross-residue bond angles (requires next residue)
            if i + 1 < len(residues): # check for next residue
                next_res = residues[i + 1] # get the next residue 
                amap_next = {a.name:a.index for a in next_res.atoms} # If there's a next residue, create its atom index map
                if "P" in amap_next: 
                    P_next = traj.xyz[:, amap_next["P"], :] # extract coordinates of P atom in next residue across 
                    for j in range(nF): # For each frame compute the angle between the current residue's C5', C1', and the next residue's P
                        angle = compute_angle(C5[j], C1[j], P_next[j]) # calculate and store the C5'-C1'-P angle across the backbone
                        if not isinstance(bond_angle[(rt, "C5", "C1", "P")], list): # if this is the first time seeing this key, initialize it as a list
                            bond_angle[(rt, "C5", "C1", "P")] = [bond_angle[(rt, "C5", "C1", "P")]] # store the angle in the dictionary under key (residue_type, "C5", "C1", "P")
                        bond_angle[(rt, "C5", "C1", "P")].append(angle) # append the angle to the list 

                        # calculate and store the X1-C1'-P angle across the backbone, same procedure as before 
                        if base_coms: # if a base COM was successfully computed for this residue
                            tag1, X1 = base_coms[0] # retrieve X1
                            angle2 = compute_angle(X1[j], C1[j], P_next[j])
                            if not isinstance(bond_angle[(rt, tag1, "C1", "P")], list):
                                bond_angle[(rt, tag1, "C1", "P")] = [bond_angle[(rt, tag1, "C1", "P")]] 
                            bond_angle[(rt, tag1, "C1", "P")].append(angle2)

                        # calculate and store the X2-X1-P angle across the backbone, same procedure as before
                        if len(base_coms) == 2: # if this is a purine residue and a 2nd base bead (X2) was computed
                            tag2, X2 = base_coms[1]
                            angle3 = compute_angle(X2[j], X1[j], P_next[j])
                            if not isinstance(bond_angle[(rt, tag1, tag2, "P")], list):
                                bond_angle[(rt, tag1, tag2, "P")] = [bond_angle[(rt, tag1, tag2, "P")]]
                            bond_angle[(rt, tag1, tag2, "P")].append(angle3)

                # calculate cross-residue angle C1-P-O5'
                if "O5'" in amap_next: # check whether the next residue has the O5' atom
                    C1_this = traj.xyz[:, amap["C1'"], :] # extract coordinates of this residue's C1 across all frames
                    P_next = traj.xyz[:, amap_next["P"], :] # next residue P
                    O5_next = traj.xyz[:, amap_next["O5'"], :] # next residue O5'
                    for j in range(nF): # For each frame j, compute the bond angle at P between atoms C1'-P-O5'
                        angle = compute_angle(C1_this[j], P_next[j], O5_next[j])
                        key = (rt, "C1", "P", "O5") # define a key for storing this angle in the bond_angle dictionary
                        if not isinstance(bond_angle[key], list):
                            bond_angle[key] = [bond_angle[key]]
                        bond_angle[key].append(angle) # store C1'-P-O5' angle from current to next residue

    # convert all accumulated values from lists into averages
    for k, v in bond_len.items(): # loop through every entry in the bond_len dictionary.
        # k is a tuple like ("A", "C1", "A1"), representing a bond between two beads in a particular residue type.
        # v is either: A single float value (already processed constant from bond table) or a list of distances, Any entry that came from dynamic measurement (not hardcoded) is a list of distances
        if isinstance(v, list): # distinguishes between the two v cases, float or list
            bond_len[k] = float(np.mean(v)) # for each bond key whose value is a list of distances: Compute the mean bond length across all frames, convert the result to a float, replace the list with this averaged value
    
    # same procedure as bond lengths
    for k, v in bond_angle.items():
        if isinstance(v, list):
            bond_angle[k] = float(np.mean(v))

    return bead_orders, bond_len, bond_angle

# final function, just gathers all the bond lengths and angles and converts them into a nice format
def geometry_parameters(bead_orders, bond_length_constants, bond_angles_constants):
    """
    Returns:
        bead_orders: a dictionary mapping residue types (e.g. "A", "G") to bead names (e.g. ["P", "O5", ...])
        bond_len: dictionary mapping (residue type, bead1, bead2) to bond lengths (in angstroms)
        bond_angle: dictionary mapping (residue type, bead1, bead2, bead3) to bond angles (in degrees)

    This version uses only predefined constant tables. It does NOT scan trajectories.
    """
    # Unit conversion helper functions
    A_to_nm = lambda x: x * 0.1 # converts angstroms to nanometers
    deg_to_rad = lambda x: np.radians(x) # converts degrees to radians

    # empty dictionaries to hold converted bond lengths and angles
    bond_len = {}
    bond_angle = {}

    # process bond lengths
    for (res, b1, b2), val in bond_length_constants.items(): # iterate over all (residue, bead1, bead2) value pairs in the provided bond lengths
        if res == "*": # if the entry uses "*" as a wildcard residue type, it applies to all residues
            for r in bead_orders: # For each nucleotide type r (e.g. "A", "G")
                bond_len[(r, b1, b2)] = A_to_nm(val) # assign the bond length value from the bond length dictionary to (r, b1, b2) after converting to nanometers
        else:
            bond_len[(res, b1, b2)] = A_to_nm(val) # If a specific residue is given (e.g., "G"), assign the converted bond length for that residue only

    # process bond angles, same as bond lengths
    for (res, b1, b2, b3), val in bond_angles_constants.items(): # iterate over the bond angle constants
        if res == "*": # If the angle entry uses "*" as a wildcard, apply it to all residue types 
            for r in bead_orders:
                bond_angle[(r, b1, b2, b3)] = deg_to_rad(val) # and convert degrees to radians.
        else:
            bond_angle[(res, b1, b2, b3)] = deg_to_rad(val)

    return bead_orders, bond_len, bond_angle

def cg_save_xyz(generated_path, filename, topology_file=None):
    """
    Save coarse-grained trajectory as an XYZ file.

    Parameters:
    - generated_path: Tensor of shape (1, T, F)
    - filename: output XYZ filename
    - topology_file: Path to a topology file (.prmtop, .pdb) used to generate bead names
    """
    # Remove batch dimension: (1, T, F) = (T, F)
    path = generated_path.squeeze(0)

    # Extract information and calculate number of beads 
    T, F = path.shape # extract T (frames) and F (flattened coordinate length).
    assert F % 3 == 0, "Each bead must have 3 coordinates (x, y, z)" # check that F is divisible by 3 (x, y, z for each bead).
    n_beads = F // 3 # calculate number of beads by dividing the number of coordiantes (features) by 3 as each bead should have 3 coordinates

    xyz = path.reshape(T, n_beads, 3) # Reshape into 3D tensor: (T, n_beads, 3) for XYZ writing.
    coordinates = xyz.cpu().numpy() # move to cpu and convert to a numpy array for easier manipulation

    # Generate bead names from topology 
    if topology_file is None: # check that topology file has been given
        raise ValueError("Either bead_names or topology_file must be provided.")

    top = md.load_prmtop(topology_file) # load topology file
    base_seq = [res.name.strip()[0].upper() for res in top.residues] # extract first character of each residue name, e.g., "G" from "GUA", "A" from "ADE"

    # define the expected CG bead layout for each nucleotide:
    base_labels_per_residue = {
        'purine': ['P', "O5'", "C5'", "C4'", "C1'", 'B1', 'B2'],
        'pyrimidine': ['P', "O5'", "C5'", "C4'", "C1'", 'B1']
    }

    bead_names = [] # list setup to store bead names
    for nt in base_seq: # loop over each base in nucleotide sequence extracted from topology file
        bead_names.extend(base_labels_per_residue['purine' if is_purine(nt) else 'pyrimidine']) # append list of bead labels to bead_names, depending on whether it's a purine or pyrimidine

    # check that the generated bead label list matches the number of CG beads
    assert len(bead_names) == n_beads, f"Expected {n_beads} bead names, got {len(bead_names)}." 

    # write to XYZ file
    with open(filename, 'w') as f:
        for frame_idx in range(T): # for each frame in the path
            f.write(f"{n_beads}\n") # write number of beads
            f.write(f"Frame {frame_idx+1}\n") # write the frame number
            for bead_idx in range(n_beads): # for each bead in the frame 
                x, y, z = coordinates[frame_idx, bead_idx] # extract coordinates from a frame and bead using the frame and bead index
                f.write(f"{bead_names[bead_idx]} {x:.5f} {y:.5f} {z:.5f}\n") # write bead label and xyz coordinates on a new line, the .5f is python string formatting to only show 5 decimal places

# function for calculating centre of mass
def compute_com(coords, elements, masses):
    """
    Compute center of mass from coordinates and atom types.
        - coords = A NumPy array of shape (n_atoms, 3 coordinates), each row is the 3D coordinate [x, y, z] of one atom
        - elements = A list of N atomic element symbols in the same order as the coordinates
    """
    w = np.asarray([masses[e] for e in elements]) # For each atom element, look up its atomic weights from the predefined dictionary, outputs a vector of atomic masses
    return np.average(coords, axis=0, weights=w) # calculate masses using a weighted average (xyz coords are averaged using the atomic weights as weights)

def is_purine(nt):
    return nt in ['A', 'G']

def save_reconstructed_xyz(coordinates, filename, residue_types, topology_file=None):
    """
    Save reconstructed CG coordinates as XYZ using residue type info.
    
    Parameters:
    - coordinates: ndarray of shape (T, N_beads, 3)
    - filename: path to save
    - residue_types: list of single-letter residue types corresponding to CG blocks
    - topology_file: optional, not required if residue_types is provided
    """
    # extract shape 
    # T is the number of frames in the trajectory.
    # N is the total number of beads per frame.
    # _ for 3D coordinates (x, y, z).
    T, N, _ = coordinates.shape

    # Convert from nanometers to angstroms for visualisers
    coordinates = coordinates * 10.0

    # map residue types to expected CG bead names
    # purines (A, G) have 7 beads (including 2 base beads X1, X2), pyrimidines (C, U, T) have 6
    base_labels_per_residue = {
        'purine':    ['P', "O5'", "C5'", "C4'", "C1'", 'X1', 'X2'],
        'pyrimidine': ['P', "O5'", "C5'", "C4'", "C1'", 'X1']
    }

    # build the complete bead_names list by expanding each nucleotide into its respective bead labels
    # EG: ['A', 'C'] = ['P', "O5'", ..., 'X2', 'P', "O5'", ..., 'X1'].
    bead_names = [] # storage list
    for nt in residue_types: # loop over each nucleotide in each residue
        bead_names.extend(base_labels_per_residue['purine' if is_purine(nt) else 'pyrimidine']) 

    # check the number of bead names must match the number of beads per frame
    assert len(bead_names) == N, f"Expected {N} bead names but got {len(bead_names)}."

    with open(filename, 'w') as f: # write to file
        for t in range(T): # for each bead (i) in the current frame
            f.write(f"{N}\n") # total number of atoms
            f.write(f"Frame {t+1}\n") # frame number
            for i in range(N): # For each bead i in the current frame
                x, y, z = coordinates[t, i] # extract coordinates of specific frame and bead
                f.write(f"{bead_names[i]} {x:.5f} {y:.5f} {z:.5f}\n") # write XYZ coordinates up to 5 decimal places

class GaussianDiffusion(nn.Module): # defining a GaussianDiffusion class from the pytorch module, contains the forward diffusion process, noise sampling and loss prediction
    def __init__( # initialises the class, defining arguments
        self, 
        model, # the neural network model that will predict the noise 
        timesteps=1000, # number of diffusion steps (default = 1000)
        beta_schedule='linear', # the schedule for the noise variance (only option is linear, more can be added later)
        loss_type='l2' # type of loss computed
    ):
        super().__init__() # initialising the base nn.Module class from pytorch
        self.model = model # passing inputs into instance variables inside this class
        self.timesteps = timesteps
        self.loss_type = loss_type

        # Define beta schedule
        # beta scheduler is the method used to define how much noise (beta) we add at each timestep
        # beta at timestep t is the variance of the Gaussian noise added at that step, controlling how much noise is added
        # used because:
            # the early timesteps are clearer data is only slightly noisy.
            # the later ones make the model robust to heavier noise.
        if beta_schedule == 'linear': # currently only linear beta scheduler is implemented
            self.betas = torch.linspace(1e-4, 0.02, timesteps) # creates a tensor of timesteps values evenly spaced from 1e-4 to 0.02, beta starts small to larger values
        else:
            raise NotImplementedError(f"Beta schedule '{beta_schedule}' not supported.") # give error if any other scheduler is given

        # precomputing values used repeatedly during both training and sampling
        # for all timesteps compute:
            # what portion of the original data (signal) survives at each step
            # how much of the original signal survives up to step t
        # this is done as wiithout knowing how much is left, you can't reconstruct how it got there or how to bring it back, also means no need to simulate all previous timesteps
        self.alphas = 1. - self.betas # alpha = 1 - beta, for each timestep, the retention factor for signal at each step.
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # computes the cumulative product of alphas up to timestep t, representing the total signal retained up to timestep t.
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # coefficient applied to the signal, determines how much of the original input is preserved at step t
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) # coefficient applied to the noise, how much noise to add at step t

    def q_sample(self, x_start, t, noise=None): # forward diffusion process = adding noise to the original data at timestep (t)
        """
        Diffuse the data for a given number of timesteps.
        """
        if noise is None:
            noise = torch.randn_like(x_start) # generate Gaussian noise with the same shape as the data (x_start)

        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start) # Fetch ᾱ and 1-ᾱ terms for each sample in the batch based on t. Shapes are broadcast to match x_start.
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise # Returns the noisy data, denoted x_t, for timestep t.

    def _extract(self, a, t, x): # Helper function to index into a 1D tensor (a) using the batch of timestep indices t, then reshape for broadcasting.
        """
        Extract values from a for batch of indices t and reshape to x_shape. This is done in order to fetch precomputer cumulative alpha values according to randomly chosen t
            - a is a 1D tensor of shape (T,), where T = number of timesteps.
            - t is a batch of sampled time indices, shape (B,), where B = batch size.
        """
        return a.to(t.device).gather(0, t).view(-1, *[1]*(x.dim()-1)) # a.to(t.device) moves a to the same device as t, then pick out from a the elements at the indices specified by t, then finally reshapes the (B,) tensor into something broadcastable to the shape of input x_t.
    
    @torch.no_grad() # disable gradient calculation during inference 
    def sample(self, model, start, end, frames, device='cuda'): # reverse diffusion process, sampling from noise to a denoised sample (this is the generative part)
        """
        Performs reverse diffusion to generate new paths from pure noise, conditioned on start and end conformations

        Args:
            model: the model predicting noise
            start, end:  conditioning information (start and end conformations for molecules)
            frames: the number of intermediate time steps you want in the generated sample
            device: run on CPU or GPU

        """
        B, feature_dim = start.shape # B = batch size (as model was trained in batches so need to tell model batch info), feature_dim = the number of features (atoms × 3 coordinates).
        T = frames # number of timesteps the model needs to generate

        # Initial noise trajectory: (B, T, F)
        # Initializes a noisy trajectory x for a starting point
        x = torch.randn(B, T, feature_dim).to(device)

        # Reverse diffusion loop
        for t in reversed(range(self.timesteps)): # Iterate from T to 0, (i.e., from noise to denoised trajectory)
            t_batch = torch.full((B,), t, dtype=torch.long).to(device) # creates a batch of the current timestep t
            t_broadcast = t_batch[:, None].expand(B, T) # expands to match the number of frames (T) so the model can condition on time for every frame.

            # Predict noise for the current state x, given current timestep and conditioning on start, end.
            pred_noise = model(x, t_broadcast, start, end)

            # If the predicted noise shape doesn't match x along the time axis, interpolate it
            # prevents mismatches due to rounding in previous operations like downsampling/upsampling.
            if pred_noise.shape[1] != x.shape[1]:
                pred_noise = F.interpolate(
                    pred_noise.permute(0, 2, 1),  # (B, F, T')
                    size=x.shape[1],              # T
                    mode='linear',
                    align_corners=False
                ).permute(0, 2, 1)               # back to (B, T, F)

            # retrieve the precomputed scalar coefficients needed for the reverse diffusion equation at timestep t
            # These control the noise schedule and how much signal vs. noise is retained or added at each timestep
            beta_t = self.betas[t] # beta_t: noise added at this step
            alpha_t = self.alphas[t] # 1 - beta_t, the retained signal
            alpha_bar_t = self.alphas_cumprod[t] # cumulative product of alphas (how much total signal remains)
            alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0).to(device) # cumulative alpha for previous step (used for noise scaling)
            # combine current signal and predicted noise to denoise the sample one step backward, derived from the DDPM reverse step equation
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            # updates x with the predicted noise, main denoising update step
            x = coef1 * (x - coef2 * pred_noise)

            # Adds new noise at timestep t if not at the final step (t = 0)
            if t > 0: 
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
                x += sigma_t * noise

        return x

    def forward(self, x, model, mask=None, start=None, end=None): # the training pass for the diffusion model.
        """
        Training loss computation.

        Args:
            x: (B, T, F) = path
            model: the UNet model
            mask: (B, T) = binary mask (1 for valid steps, 0 for padding)
        """
        B, T, F = x.shape # Unpacks batch dimensions and device info.
        device = x.device 

        t = torch.randint(0, self.timesteps, (B,), device=device).long() # sample a random timestep t for each sample in the batch
        noise = torch.randn_like(x) # make noise to add

        x_t = self.q_sample(x, t, noise=noise) # Computes the noisy version of the data at timestep t using the q_sample function

        # reshapes t to (B, T) so that it matches the shape of the data it's associated with (B, T, F)
        t_broadcast = t.view(B, 1).expand(B, T)  # Expands the timestep tensor t to shape (B, T) so that all frames in the batch have the same timestep (t)

        # Use provided start/end, if not provided, then just extract from the path itself
        if start is None:
            start = x[:, 0]  # (B, F)
        if end is None:
            end = x[:, -1]  # (B, F)

        # Model prediction (typically noise)
        pred_noise = model(x_t, t_broadcast, start = start, end = end)  # start = x[:, 0], end = x[:, -1]
        # feeds noisy data to the model
        # Also gives:
            #t_broadcast: timestep for conditioning,
            #x[:, 0]: start frame (shape (B, F)),
            #x[:, -1]: end frame (shape (B, F)).

        # shape mismatches can occur in Unet architectures due to upsampling and downsampling as they scale by 2, so if inputs/outputs are not powers of 2 rounding can give mismatched values
        if pred_noise.shape != noise.shape: # check if data shapes match as it can error out otherwise
            min_T = min(pred_noise.shape[1], noise.shape[1]) # Computes the minimum valid size along the time dimension T and feature dimension F.
            min_F = min(pred_noise.shape[2], noise.shape[2])
            pred_noise = pred_noise[:, :min_T, :min_F] # Trims both pred_noise and noise tensors so they match exactly in shape.
            noise = noise[:, :min_T, :min_F]
            if mask is not None: # adjusts the mask to match.
                mask = mask[:, :min_T]

        if self.loss_type == 'l2': # only one type of loss is currently supported
            loss = (pred_noise - noise) ** 2 # Computes element-wise squared error between predicted noise and the ground-truth noise.
        elif self.loss_type == 'periodic':
            loss = periodic_l2_loss(pred_noise, noise)
        else:
            raise NotImplementedError(f"Loss type '{self.loss_type}' not supported.")
        
        if mask is not None: # if mask is given
                mask = mask.unsqueeze(-1)  # expands mask to match last dimension
                loss = loss * mask # Applies it to the loss to ignore padded frames.
                loss = loss.sum() / mask.sum().clamp(min=1.0) # Computes the masked mean loss.
        else:
            loss = loss.mean() # If no mask is given, just take the average loss over all elements.

        return loss # Returns the final computed loss to be used for backpropagation during training.
    
#building a dataset class to handle path data
#uses the base pytorch dataset utility to build
#import relevant libraries

class MolecularPathDataset(Dataset):
    def __init__(self, data): # instantiating the Dataset object, initialize with the frame data, required by pytroch
        """
        Args:
            data: A list of tuples in the form (start_frame, end_frame, path),
                         where:
                           - start_frame: 1D numpy array of shape (n_atoms * 3,)
                           - end_frame: 1D numpy array of shape (n_atoms * 3,)
                           - path: 2D numpy array of shape (n_frames, n_atoms * 3)
        """
        self.data = data # store the list of (start, end, path) tuples inside the class for later access.

    def __len__(self): # how many samples are in the dataset, allows len(dataset) to work and also enables batching in the DataLoader, standard requirement from Pytorch
        return len(self.data) # returns the number of trajectories (i.e., how many (start, end, path) triplets are stored).

    def __getitem__(self, idx): # retrieves a single item from the dataset at a certain index, allowing retrieval of data from the dataset class
        start, end, path = self.data[idx] # unpacks the tuple stored at index idx

        # convert each numpy array to a PyTorch tensor
        start_tensor = torch.from_numpy(start).float() # the first frame (1D vector)
        end_tensor = torch.from_numpy(end).float() # the last frame (1D vector)
        path_tensor = torch.from_numpy(path).float() # the full sequence of frames (2D array: frames, flattened coordinates)

        return { # returns a dictionary of start, end and path data to allow for easy access from the dataset
            'start': start_tensor,      # shape: (n_atoms * 3,) 
            'end': end_tensor,          # shape: (n_atoms * 3,)
            'path': path_tensor         # shape: (n_frames, n_atoms * 3)
        }

# machine learning models cannot handle dynamic length inputs, need to pad the frames out so they match
# function to pad frame data so number of frames across each path matches
# this function is meant to be used in the dataloader from PyTorch so it can handle batches of data
# each batch is a list of individual samples, where each sample comes from MolecularPathDataset.__getitem__() and is a dictionary with keys 'start', 'end', and 'path'

def collate_paths(batch): # function to pad frames data to number of frames across paths matches and doesnt cause errors in the model
    """
    Custom collate function for batching variable-length molecular paths.
    B = batch size, T = number of time steps (padded), F = flattened frame size (i.e n_atoms * 3)

    Args:
        batch (list): List of samples from MolecularPathDataset.
                      Each sample is a dict with keys 'start', 'end', 'path'.

    Returns:
        A dictionary with: 
            - 'start': (B, F) tensor of start frames
            - 'end': (B, F) tensor of end frames
            - 'path': (B, T, F) padded tensor of paths
            - 'mask': (B, T) tensor where 1 indicates valid step, 0 is padding
    """
    # Collect paths, starts, and ends
    paths = [sample['path'] for sample in batch]         # grab all the path tensors in the batch
        #Each 'path' is a 2D tensor with shape (n_frames, n_atoms * 3), and these can vary in path length (n_frames).
    starts = torch.stack([sample['start'] for sample in batch])  # grab all start tensors/data from the batch and stack into a 2D tensor (shape = (B, F))
    ends = torch.stack([sample['end'] for sample in batch])      # grab all end tensors/data from the batch and stack into a 2D tensor (shape = (B, F))

    # Pad paths to max length in batch with zeroes
    padded_paths = pad_sequence(paths, batch_first=True)  # pad_sequence function from pytorch used to pad path lengths to the length of the longest one in the batch
    # outputs a 3D tensor of shape (B, T_max, F), where T_max is the longest path in the batch.

    # create a mask tensor filled with zeros to track which values are padding vs. actual data (1 for valid timesteps, 0 for padding)
    mask = torch.zeros(padded_paths.shape[:2], dtype=torch.float32)  # (batch size, max path length in the batch)
    for i, p in enumerate(paths): # for path (i) in the batch
        mask[i, :p.shape[0]] = 1.0 # sets the number of actual frames(p.shape[0]) in row i of the mask to 1, indicating real frames, remaining padded frames are 0
        # i = the index of the current sample in the batch
        # p = a path tensor (shape (n_frames, n_atoms * 3)) from the paths list, getting the shape will tell the length of the path which is recorded in the mask

    return { # Returns a dictionary 
        'start': starts,             # batch of start vectors (batch size, frame data)
        'end': ends,                 # batch of end vectors (batch size, frame data)
        'path': padded_paths,        # padded batch of full sequences (batch size, max path length in the batch, frame data)
        'mask': mask                 # same shape as path length, tells your model which parts are real data (batch size, max path length in the batch)
    }

class EMA: # implement Exponential Moving Average
    #keeps a shadow version of the model with slowly updating weights:
        # at every training step, instead of replacing the weights completely, blend the current weights with the old ones.
            # this helps smooth out noisy updates as gradient updates can massively change during training, thus making EMA models less sensitive to local noise and more accurately representing overall trends
            # by blending weights it helps resist overfitting in later stages of training from the main model as it will not immediately follow the same behaviour
            # ema_decay is the weighting factor that determines how much of the previous EMA value is kept vs. how much of the current model weight is incorporated
        # results in a model that evolves more slowly and generally performs better at inference time.
            # thus at inference time, use the EMA model instead of the current model.
    def __init__(self, model, decay): # class initiator/constructor, defines arguments (model is original model whose weights are tracked)
        self.ema_model = deepcopy(model) # make a full copy of the model. ema_model will store the EMA-smoothed weights.
        self.decay = decay # stored float between 0 and 1 controlling how slow the EMA updates (0.9 = slower changes)
        self.ema_model.eval() # Sets the EMA model to evaluation mode to disable updates
        self.ema_model.requires_grad_(False) # freeze parameters ensuring the EMA model is not trained with gradients.

    def update(self, model): # used to update EMA model with current model during training
        with torch.no_grad(): # manual weight update, no gradients needed
            for ema_param, param in zip(self.ema_model.parameters(), model.parameters()): # Iterate over the pairs of parameters from the EMA model and the actual model.
                ema_param.data.mul_(self.decay).add_(param.data, alpha=1.0 - self.decay) # update EMA model 

    def state_dict(self): # Returns the weights of the EMA model (to save or load it separately).
        return self.ema_model.state_dict()
    
    def to(self, device): # Moves the EMA model to the specified device (GPU or CPU).
        self.ema_model.to(device)

# Trainer class setup to handle the training loop diffusion model, including gradient scaling, EMA tracking, and saving final models.
class Trainer:
    def __init__( # setup
        self,
        diffusion, # the diffusion model wrapepr, containing the Unet model
        dataloader, # dataloader containing dataset
        ema_decay=0.995, # weighting factor that determines how much of the previous EMA value you keep vs. how much of the current model weight you incorporate (0.995 means keep 99.5% of EMA weight, only mix in 0.5% of the new model weight.)
        learning_rate=1e-4, # model learning rate
        device=None,
        results_folder='./results', # folder to save model to
        save_name='model-final.pt', # name of model
        use_amp=False # whether or not to use automatic mixed precision
    ):
        self.diffusion = diffusion # store diffusion model
        self.model = diffusion.model # extract underlying Unet
        self.dataloader = dataloader # store dataloader 
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu') # train on GPU if available, if not then CPU
        self.model.to(self.device) # Move both the model to the selected device.
        self.diffusion.to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate) # Use ADAM optimiser for model training, can change later
        self.scaler = torch.amp.GradScaler(device=self.device, enabled=use_amp) # if use_amp is True, then use automatic mixed precision support from pytorch(faster + less memory)

        self.ema = EMA(self.model, decay=ema_decay) # initialise EMA copy of model with selected decay
        self.ema.to(self.device) # move model to device

        self.results_folder = results_folder
        os.makedirs(results_folder, exist_ok=True) # check if results folder exists
        self.save_path = os.path.join(results_folder, save_name) # get full path of results folder

        self.use_amp = use_amp 

    def train(self, num_epochs): # the actual training loop
        self.model.train() # sets model to training mode
        for epoch in range(num_epochs): # iterates over number of epochs
            pbar = tqdm(self.dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}") # wrap the dataloader with a tqdm progress bar.
            for batch in pbar: # Move each component of the batch to the GPU: starting frame, ending frame, path sequence, and mask.
                start = batch['start'].to(self.device)           # (B, F)
                end = batch['end'].to(self.device)               # (B, F)
                path = batch['path'].to(self.device)             # (B, T, F)
                mask = batch['mask'].to(self.device)             # (B, T)

                self.optimizer.zero_grad() # Clear previous gradients.

                with torch.autocast(device_type= (self.device or 'cuda' if torch.cuda.is_available() else 'cpu'), enabled=self.use_amp): # If AMP is enabled, compute the loss in mixed precision.
                    loss = self.diffusion(path, self.model, mask=mask, start = start, end = end) # call diffusion model forward function 

                self.scaler.scale(loss).backward() # Scale the loss to prevent underflow (AMP trick).
                self.scaler.step(self.optimizer) # Perform backward pass and optimizer step.
                self.scaler.update() # Update the gradient scaler for the next iteration.

                self.ema.update(self.model) # update EMA model
                pbar.set_postfix(loss=loss.item()) # Show the current loss in the progress bar.

        # Save final model and EMA model
        torch.save({
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, self.save_path)
        print(f"Model saved to {self.save_path}")

# Unet setup

# Sinusoidal Positional Embedding
# in diffusion models, timestep embeddings tell the model where in the denoising process it is (e.g., step 5 vs. step 999). 
# Instead of just passing the integer t, transform it into a continuous vector using sine/cosine patterns.
# Sinusoidal positional embeddings encode time or position information using a combination of sine and cosine functions at different frequencies
    # allows each timestep to be represented uniquely, and the model can learn to associate patterns across time.
class SinusoidalPosEmb(nn.Module): # define class/module
    def __init__(self, dim): # dim = size of embedding vector produced
        super().__init__()
        self.dim = dim

    def forward(self, t): # forward pass, t is a tensor of shape (B,) one timestep per batch element (usually int64 or float32).
        device = t.device # move to selected device
        half_dim = self.dim // 2 # split the full dimension in half because we compute sine for the first half, and cosine for the second half.
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / half_dim)) # Create a tensor of size half_dim with exponentially scaled values to control sine/cosine frequency 
        emb = t.unsqueeze(-1) * emb # turns (B,) into (B, 1) so it can be multiplied element-wise with the emb vector, giving tensor of shape (B, half_dim)
        # Each row is now a vector of scaled time values for sine and cosine functions.
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # take the sine of the first half and cosine of the second half and concatenate the two halves
        return emb # results in a smooth, continuous embedding of the timestep that carries both high-frequency and low-frequency information  helping the model know how far along the diffusion process is.

# Residual Block with time embedding
# A ResNet (short for Residual Network) is a neural network architecture built out of many stacked residual blocks. 
# this is a mini-residual block adapted for 1D convolutional architecture
class ResnetBlock(nn.Module): # ResNet block for Unet made from pytorch class
    def __init__(self, dim_in, dim_out, time_emb_dim, groups=8): 
        """
        dim_in = Number of input channels
        dim_out = Number of output channels
        time_emb_dim: Dimensionality/size of the time embedding vector.
        groups: Number of groups for GroupNorm 
        """
        super().__init__()

        # multi layer perceptron
        # Takes a time embedding vector (e.g., from sinusoidal encoding).
        # Passes it through a nonlinear transformation.
        # Outputs a vector that can be added to the convolutional feature map, aligning the temporal signal with spatial features.
        self.mlp = nn.Sequential( # Projects the t_emb vector (sinusoidal timestep embedding) through a nonlinearity and a linear layer, to match the shape of the feature map (dim_out).
            Mish(), 
            nn.Linear(time_emb_dim, dim_out)
        )
        # normalization layers
        # GroupNorm normalizes across groups of channels instead of across the batch, helps stabilize training by normalizing features in channel groups 
            # splits channels into groups of 8 and normalizes within each group.
        self.norm1 = nn.GroupNorm(groups, dim_in) 
        self.norm2 = nn.GroupNorm(groups, dim_out)
        self.act = Mish() # apply activation function

        # main feature extraction by 2 1D convolutional layers with kernel size 3 and padding 1 so the output tensor after the convolution should still have the same length T in the time dimension.
        self.conv1 = nn.Conv1d(dim_in, dim_out, 3, padding=1)
        self.conv2 = nn.Conv1d(dim_out, dim_out, 3, padding=1)

        # Residual connection setup
        # If input and output dims don't match, apply a 1D conv to match them.
        # Otherwise, pass input as an normal identity function (nn.Identity()) 
        self.residual = nn.Conv1d(dim_in, dim_out, 1) if dim_in != dim_out else nn.Identity()

    # forward pass
    # x: Input feature map of shape (B, C, T) (Batch, Channels, Time)
    # t_emb: Timestep embedding vector (from sinusoidal)
    # pre activation (Normalization -> Activation -> Convolution) is used
    def forward(self, x, t_emb):
        h = self.norm1(x) # normalize input
        h = self.act(h) # apply activation function
        h = self.conv1(h)# 1st convolution (projects input from dim_in to dim_out), transforms x into a richer feature representation.

        t_out = self.mlp(t_emb).unsqueeze(-1) # Apply the multi layer preceptron to the time embedding. unsqueeze(-1) reshapes it to match shape (B, C, 1) for broadcasting.
        h = h + t_out # Add it to the hidden feature map, this injects timestep information into the block and adds temporal context into each time slice of the sequence.

        h = self.norm2(h) # Normalize
        h = self.act(h) # Activation function 
        h = self.conv2(h) # 2nd convolution

        return h + self.residual(x) # Final output is the refined path (h) plus a shortcut version of the original input (x), the residual connection

# Downsample and Upsample layers
class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)

# Mish activation function 
class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))
    
# Residual with Zero Initialization
# A residual connection is a shortcut path in a neural network that bypasses one or more layers and adds the input directly to the output of those layers. In its simplest form: output=x+f(x)
# x is the input to a block of layers, f(x) is the output of the block usually some form on non-linear transformation (convolution layers, etc.), x+f(x) is the final output: the residual connection.
# ReZero changes this to output = x + g * f(x), where g is a learnable scalar parameter, initialized at 0
# As training progresses, alpha learns how much of f() should be added.
class Rezero(nn.Module):
    def __init__(self, fn): # single argument fn, which is the function (usually another module like an attention block) that will be wrapped.
        super().__init__()
        self.fn = fn # stores the function internally
        self.g = nn.Parameter(torch.zeros(1))  # learnable scale initialized at 0
        # self.g is a learnable scalar parameter starting at 0, equation wise it looks like: output = x + 0 * fn(x)
        # During training, it will gradually grow, allowing the function fn(x) to contribute more over time, essentially g learns how much of attn(x) to inject into the output.

    def forward(self, x):
        return x + self.g * self.fn(x) # Computes the output of the wrapped function: fn(x), scales it by g, adds it to the input x, creating a residual connection like in ResNets.
    
# Linear Attention class
class LinearAttention1d(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        """
        dim: input channel dimension (number of features)
        heads: number of attention heads (default 4)
        dim_head: size of each head’s projection (default 32)
        """
        super().__init__() # initializes nn.Module base class
        self.heads = heads
        inner_dim = dim_head * heads # Total inner dimension for query/key/value is dim_head * heads
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False) # Applies a 1D convolution to project input x into Q, K, and V, each of shape (inner_dim, t)
                                                                   # Since we need 3 sets (Q, K, V), the output has size 3 * inner_dim.
        self.to_out = nn.Conv1d(inner_dim, dim, 1) # Projects the attention result back to the original input dimension with another 1D conv.

    def forward(self, x):
        """
        x: input tensor of shape (batch, channels, time)
        """
        b, c, t = x.shape # B = batch size, F = number of channels/features, T = sequence length
        qkv = self.to_qkv(x).chunk(3, dim=1) # Applies the convolution to compute Q, K, and V in one go
        q, k, v = map(lambda x: x.reshape(b, self.heads, -1, t), qkv) # splits the tensor into 3: q, k, v, each shape (b, inner_dim, t) and reshape each tensor to: (batch, heads, dim_head, time)
        # this makes each head independent for multi-head attention

        # softmax over sequence length
        q = q.softmax(dim=-1)
        k = k.softmax(dim=-2)

        # context computation
        context = torch.einsum('bhdt,bhds->bhds', k, v) # Sums the product of the elements of the input operands along dimensions specified using a notation based on the Einstein summation convention
        # For each time point, aggregates the value vectors, weighted by keys

        out = torch.einsum('bhdt,bhds->bhdt', q, context) # Applies attention: multiply softmaxed queries q by the precomputed context

        out = out.reshape(b, -1, t) # Reshape from (batch, heads, dim_head, time) -> (batch, inner_dim, time)
        return self.to_out(out) # Final convolution maps back to original input channel size

# UNet1D with down/up path and ResNet blocks
class UNet(nn.Module):
    def __init__(self, input_dim, base_dim=64, dim_mults=(1, 2, 4), time_emb_dim=128, out_dim=None):
        """
        Args:
            input_dim: input feature dimension (number of atoms * 3 coordinates)
            base_dim: base number of channels (number of atoms)
            dim_mults: how channels increase at each level.
            time_emb_dim: size of time embedding vector from Sinusoidal
            out_dim: final output dimension (defaults to input_dim if not set)
        """
        super().__init__() # initialize pytorch base class

        self.time_mlp = nn.Sequential( # time embedding multi layer perceptron
            SinusoidalPosEmb(time_emb_dim), # Sinusoidal timestep embedding mapping scalar t into a vector with sine/cosine functions 
            # helps the model distinguish between different timesteps using smooth, periodic signals
            nn.Linear(time_emb_dim, time_emb_dim * 4), # projects the sinusoidal embedding up to a larger dimensional space.
            Mish(), 
            nn.Linear(time_emb_dim * 4, time_emb_dim) # brings the dimensionality back down to the original time_emb_dim
            # outputs vector representation of the timestep that can be injected into the model 
        )

        self.se_mlp = nn.Sequential( # multi layer perceptron for adding start and end conformations into the timestep embedding
        nn.Linear(input_dim * 2, time_emb_dim), # takes a concatenated input of start and end data and compresses the data
        Mish(),
        nn.Linear(time_emb_dim, time_emb_dim) # output final vector representation of the start and end frames
        )

        dims = [base_dim * m for m in dim_mults] # calculate channel dimensions (if base_dim = 64 and dim_mults = (1, 2, 4), then dims = [64, 128, 256])
        self.input_proj = nn.Conv1d(input_dim * 3, dims[0], 1) # Projects input from input_dim to first hidden dim using a 1D conv with kernel size 1.

        # Lists to store layers in the downsampling and upsampling paths, this allows modular number of layers depending on inputs
        self.downs = nn.ModuleList() 
        self.ups = nn.ModuleList()

        # Downsampling path
        for i in range(len(dims) - 1): # iterate for every dim, allowing modular number of layers in the Unet
            in_dim = dims[i + 1]
            self.downs.append(nn.ModuleList([ # add to list of layers
                ResnetBlock(dims[i], in_dim, time_emb_dim), # ResnetBlock increases feature richness.
                ResnetBlock(in_dim, in_dim, time_emb_dim),
                Rezero(LinearAttention1d(in_dim, heads=4, dim_head=in_dim // 4)), # use ReZero and attention, also set dim_head such that the total inner dimension of the attention equals the input channels.
                Downsample(in_dim) # A Downsample layer halves the temporal resolution.
            ]))
        
        # Middle block
        # Two ResNet blocks at the deepest level of the UNet
        # Fully retains time conditioning.
        self.mid_block1 = ResnetBlock(dims[-1], dims[-1], time_emb_dim)
        self.mid_attn = Rezero(LinearAttention1d(dims[-1], heads=4, dim_head=dims[-1] // 4))
        self.mid_block2 = ResnetBlock(dims[-1], dims[-1], time_emb_dim)

        # Upsampling path
        for i in reversed(range(len(dims) - 1)): # iterate for every dim, allowing modular number of layers in the Unet
            in_dim = dims[i]
            self.ups.append(nn.ModuleList([ # add to list of layers
                ResnetBlock(dims[i + 1] * 2, in_dim, time_emb_dim),
                ResnetBlock(in_dim, in_dim, time_emb_dim),
                Rezero(LinearAttention1d(in_dim, heads=4, dim_head=in_dim // 4)), 
                Upsample(in_dim)
            ]))

        self.output_block = ResnetBlock(dims[0], dims[0], time_emb_dim) # final ResNet block
        self.output_proj = nn.Conv1d(dims[0], out_dim or input_dim, 1) # projection back to original feature dimension with a final convolutional layer

    def forward(self, x, t, start, end):
        """
        Args:
            x: input trajectory — shape (B, T, F)
            t: timestep tensor — shape (B, T)
        """
        start = start.to(x.device)
        end = end.to(x.device)
        # expand the start and end frames so that they match the shape of the path data, this is for concatenation later 
        start = start.unsqueeze(1).expand(-1, x.shape[1], -1)
        end = end.unsqueeze(1).expand(-1, x.shape[1], -1)

        # concat start and end along feature dimension (F) of the frame data
        # The current noised frame (x)
        # The fixed start frame (start)
        # The fixed end frame (end)
        # start and end are conditioning information, adding context to information so model has starting and end points to move towards
        x = torch.cat([x, start, end], dim=-1)  # new shape: (B, T, F*3) 

        # Convert input from (B, T, F) to (B, F, T) for 1D convolutions (PyTorch expects channels first).
        x = x.permute(0, 2, 1)
        t_emb = self.time_mlp(t[:, 0])  # Apply time embedding MLP to the first timestep of each batch sample (B,) -> (B, time_emb_dim)

        # embed start and end using a small MLP
        start_global = start[:, 0, :] # split out the expanded start/end frames as no need to include time dimension (B,T,F) -> (B,F)
        end_global = end[:, 0, :]    
        se_emb = torch.cat([start_global, end_global], dim=-1)  # concatenate the start and end frames together into shape (B, 2F)
        se_emb = self.se_mlp(se_emb)             # apply the earlier multi layer perceptron outputs: (B, time_emb_dim)
        t_emb = t_emb + se_emb                   # put start/end into timestep embedding so model understands what the start and end point in time looks like, as meaning of time depends on what the start and end points are.

        # Projects input features and stores for skip connections.
        x = self.input_proj(x)
        h = [x]

        # Down path
        for resnet, resnet2, attn, down in self.downs:
            x = resnet(x, t_emb)
            x = resnet2(x, t_emb)
            x = attn(x)
            h.append(x)
            x = down(x)

        # Middle
        # Process through two residual layers at the bottom of the UNet
        x = self.mid_block1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t_emb)

        # Up path
        for resnet, resnet2, attn, up in self.ups:
            skip = h.pop() # Get skip connection from encoder.
            # Fix any small mismatches in temporal length due to rounding in downsampling as both must tensors match in all dimensions except the channel one otherwise pytorch errors out
            if skip.shape[-1] > x.shape[-1]:
                skip = skip[..., :x.shape[-1]] # If the skip connection is longer, slice it down to match x
            elif skip.shape[-1] < x.shape[-1]:
                x = x[..., :skip.shape[-1]] # If x is longer (e.g., after upsampling),truncate it to match the skip.

            x = torch.cat((x, skip), dim=1) # # Concatenates skip connections from downsampling to be used (so input dim is doubled).
            x = resnet(x, t_emb) # Resnet block
            x = resnet2(x, t_emb)
            x = attn(x)
            x = up(x) # upsample

        # Apply final ResNet block and project back to out_dim or input_dim.
        x = self.output_block(x, t_emb)
        x = self.output_proj(x)

        # Convert back to shape (B, T, F) for compatibility with rest of the diffusion model.
        return x.permute(0, 2, 1)  # (B, F, T) -> (B, T, F)
