import torch
from Landscape_DDPM import GaussianDiffusion, UNet
import mdtraj as md
import numpy as np

# Config
topology_file = '.../coords.prmtop'  # shared topology
PO5_BOND_LENGTH = 0.1615  # nm
atomic_weights = {"C": 12.011, "N": 14.007, "O": 16.000}
CG_bases = {
    "A1": ["N7", "N9", "C4", "C5", "C8"],
    "A2": ["N1", "C2", "N3", "C4", "C5", "N6", "C6"],
    "C1": ["N1", "C2", "N3", "N4", "C4", "C5", "C6", "O2"],
    "G1": ["N7", "N9", "C4", "C5", "C8"],
    "G2": ["N1", "N2", "C2", "N3", "C4", "C5", "C6", "O6"],
    "T1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"],
    "U1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"]
}

def compute_com(coords, elements):
    weights = np.array([atomic_weights[e] for e in elements])
    return np.average(coords, axis=0, weights=weights)

# load and coarse-grainn a single path to get the start and end conformations for conditioning
traj = md.load_mdcrd('/path.mdcrd' , top=topology_file)
top = traj.topology
n_frames = traj.n_frames

cg_beads = []
# main CG processing
for res in top.residues: # loop over each residue 
    atom_map = {a.name: a.index for a in res.atoms} # for each atom in the residue build a dictonary to map atom names to their indicies used later 

    # backbone bead extraction
    if res.index == 0: # special case for 1st nucleotide: estimate P from HO5' and O5'
        ho5 = traj.xyz[:, atom_map["HO5'"], :] # extract coordinate positions of HO5'
        o5 = traj.xyz[:, atom_map["O5'"], :] # extract coordinate positions of O5'
        direction = o5 - ho5 # find direction vector between HO5' and O5'
        unit_vec = direction / np.linalg.norm(direction, axis=1, keepdims=True) # normalize the direction vector by dividing it with its length to remove the magnitude (length) of the vector so we only have the direction in the form of a unit vector
                                                                                # np.linalg.norm is used to calculate the Euclidean length of the direction vector 
        p_bead = o5 + unit_vec * PO5_BOND_LENGTH # extending vector to match P-O5' bond length
    else:
        p_bead = traj.xyz[:, atom_map["P"], :] # if is not 1st atom then just extract the P

    # extract coordinates of specific atoms for CG beads
    # outputs 5 arrays with (n_frames, 3 coordinates)
    o5_bead = traj.xyz[:, atom_map["O5'"], :] 
    c5_bead = traj.xyz[:, atom_map["C5'"], :]
    c4_bead = traj.xyz[:, atom_map["C4'"], :]
    c1_bead = traj.xyz[:, atom_map["C1'"], :]

    # collect backbone by setting up a list of 5 arrays 
    backbone = [p_bead, o5_bead, c5_bead, c4_bead, c1_bead]  # (5, n_frames, 3 coordiantes)
    
    # Base COMs
    resname = res.name.strip() # extract the name of the residue 
    base_coms = []
    # base centre of mass bead calculations
    for key, atom_list in CG_bases.items(): # extract the base names and atom lists from the defined dictionary setup before
        if key.startswith(resname[0]): # resname is the current residue's name (e.g., "ADE", "CYT"). resname[0] would be the base letter: "A", "C", "G", "T", or "U". 
                                        # checks if the CG base type (eg: "A1", "A2") matches the nucleotide
            coords_list = [] # initialize lists to store the coordinates and element names
            elements_list = []
            for atom_name in atom_list: # loop over each atom in the list of predefined CG base atoms 
                if atom_name in atom_map: # check if the name of the atom is in the atom map of the residue
                    coords_list.append(traj.xyz[:, atom_map[atom_name], :]) # extract and append the coordinates of that atom for every frame. creates a list of arrays of shape (n_frames, 3)
                    elements_list.append(top.atom(atom_map[atom_name]).element.symbol) # extract the element symbol (e.g., 'C', 'N') for that atom index and appends it to a list
            if len(coords_list) > 0: # proceed only if we successfully collected atoms from the residue
                coords_arr = np.stack(coords_list, axis=1)  # join the list of arrays together to create a shape of (n_frames, n_atoms, 3)
                coms = np.array([compute_com(coords_arr[i], elements_list) for i in range(n_frames)]) # for each frame of the path compute the COM for the nucleotide, returns an array of shape (n_frames, 3) — 1 or 2 COM per frame
                base_coms.append(coms[:, np.newaxis, :]) # save value to list
    base_coms = None
    if base_coms is not None: # if we have a base bead add it to the full CG set
        base_beads_cat = np.concatenate(base_coms, axis=1)  #  concatenate the beads together so it can be concatenated into the full bead set later
        full_bead_set = backbone + [base_beads_cat]
    else: # if no base bead is calculated it's just the backbone set
        full_bead_set = backbone

    # merge all beads into a single array per residue into shape (n_frames, N_beads, 3)
    # Each bead is a NumPy array of shape (n_frames, 3): coordinates across trajectory frames for a single bead
    reshaped_beads = [b if b.ndim == 3 else b[:, np.newaxis, :] for b in full_bead_set] # check that all bead arrays are shape (n_frames, N_beads, 3)
                                                                                        # If the bead has shape (n_frames, 3) (i.e., a single bead) insert a new axis at position 1 → becomes (n_frames, 1, 3).
    residue_beads = np.concatenate(reshaped_beads, axis=1) #  concatenates beads along axis 1 (the bead dimension), outputs (n_frames, total_beads_per_residue, 3 coordiantes)

    cg_beads.append(residue_beads) # append to list

# Join together the coarse-grained coordinates from each residue in the molecule along the bead dimension.
cg_traj = np.concatenate(cg_beads, axis=1)  # (n_frames, total_beads, 3)

# Flatten each frame from 2D (total_beads, 3) to 1D (total_beads * 3) so each frame becomes a single vector of coordinates.
cg_flat = cg_traj.reshape(n_frames, -1)  # (n_frames, total_beads * 3)

# Extract start and end structure and convert to tensors for inference
start = torch.tensor(cg_flat[0], dtype=torch.float32).unsqueeze(0)
end = torch.tensor(cg_flat[-1], dtype=torch.float32).unsqueeze(0)

# get number of beads
F = start.shape[-1]
n_atoms = F // 3

# Load model state dict
# define the U-net structure
model = UNet(
    input_dim=F,           # atoms × 3 coordinates
    base_dim=32,
    dim_mults=(1, 2, 2, 4),
    time_emb_dim=128,
    out_dim=None         # same as in_dim by default
)

# Load diffusion wrapper
diffusion = GaussianDiffusion(model, timesteps=100)
checkpoint = torch.load(".../molecular_path_diffusion_CG.pt")
model.load_state_dict(checkpoint['ema']) # load in the EMA model weights into the defined model
model = model.to('cuda') # move model to cuda, hardcoded fix
model.eval() # set to evaluate mode so weights don't change

# run inference
generated_path = diffusion.sample(model, start, end, frames=50, device='cuda')  # (1, 50, F)

# save generated path as XYZ file
from Landscape_DDPM import cg_save_xyz
cg_save_xyz(generated_path, 'generated_path_CG.xyz', topology_file=topology_file)
