import mdtraj as md
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict
import torch
from Landscape_DDPM_v2 import compute_dihedrals
from Landscape_DDPM_v2 import compute_com
# Configuration
trajectory_folder = '.../Paths/'       # folder containing all .mdcrd files
topology_file = '.../coords.prmtop'  # shared topolog
prmtop_file = '.../coords.prmtop'  # shared topology

PO5_BOND_LENGTH = 0.1615  # in nm, used to scale the O5' and OH bond in the 1st residue to a P-O5' bond length for consistency

# atomic weights for centre of mass calculations
atomic_weights = {"C": 12.011, "N": 14.007, "O": 16.000}

# nucleotide base definitions
# defines which atoms are used to compute base COMs for the base CG beads
CG_bases = {
    "A1": ["N7", "N9", "C4", "C5", "C8"],
    "A2": ["N1", "C2", "N3", "C4", "C5", "N6", "C6"],
    "C1": ["N1", "C2", "N3", "N4", "C4", "C5", "C6", "O2"],
    "G1": ["N7", "N9", "C4", "C5", "C8"],
    "G2": ["N1", "N2", "C2", "N3", "C4", "C5", "C6", "O6"],
    "T1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"],
    "U1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"]
}

# define beads used for dihedral calculations
PURINE_QUADRUPLETS = [  # For A, G (10 dihedrals)
    (3, 4, 5, 6),   # 0: R4-R1-A1/G1-A2/G2
    (3, 5, 6, 4),   # 1: R4-A1/G1-A2/G2-R1
    (2, 3, 4, 5),   # 2: C-R4-R1-X1
    (0, 3, 4, 5),   # 3: P-R4-R1-X1
    (2, 3, 0, 1),   # 4: C-R4-P-O
    (4, 3, 0, 1),   # 5: R1-R4-P-O
    (1, 2, 3, 0),   # 6: O-C-R4-P
    (1, 2, 3, 4),   # 7: O-C-R4-R1
    (0, 1, 2, 3),   # 8: P-O-C-R4
    (3, 0, 1, 2)    # 9: R4-P-O-C
]

PYRIMIDINE_QUADRUPLETS = [  # For C, T, U (8 dihedrals, skipping 0 and 1)
    (2, 3, 4, 5),   # 2: C-R4-R1-X1 
    (0, 3, 4, 5),   # 3: P-R4-R1-X1 
    (2, 3, 0, 1),   # 4: C-R4-P-O 
    (4, 3, 0, 1),   # 5: R1-R4-P-O 
    (1, 2, 3, 0),   # 6: O-C-R4-P 
    (1, 2, 3, 4),   # 7: O-C-R4-R1 
    (0, 1, 2, 3),   # 8: P-O-C-R4 
    (3, 0, 1, 2)    # 9: R4-P-O-C 
]

all_paths = [] # list to store the dihedral paths

# Process each trajectory
for filename in tqdm(os.listdir(trajectory_folder)): # iterate over every path file in the provided folder
    if not filename.endswith(".mdcrd"): # skip over files that do not end in mdcrd
        continue

    # load trajectory/path and topology of path using MDTraj
    path_file = os.path.join(trajectory_folder, filename) # build a full path file for loading function
    traj = md.load_mdcrd(path_file, top=prmtop_file) # load file
    top = traj.topology # get topology of path
    n_frames = traj.n_frames # extract number of frames in path
    
    dihedral_trajs = []  # list to store dihedrals per residue per frame

    for res in top.residues: # loop over each residue
        atom_map = {a.name: a.index for a in res.atoms} # create a dictionary of atom names to atom index in the residue
        resname = res.name.strip() # get residue name

        # backbone bead calculation
        if res.index == 0: # for the 1st residue 
            ho5 = traj.xyz[:, atom_map["HO5'"], :] # extract XYZ coordinate of the HO5
            o5 = traj.xyz[:, atom_map["O5'"], :] # extract XYZ coordinate of the O5'
            direction = o5 - ho5 # find a direction vector between these atoms, should be the bond HO5'-O5'
            unit_vec = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8) # normalize the bond vector for each frame to obtain a direction-only unit vector
                                                                                             # axis=1 computes norm along x/y/z dimensions
                                                                                             # + 1e-8 avoids division-by-zero
            p_bead = o5 + unit_vec * PO5_BOND_LENGTH # from the O5' atom scale the bond length in a direction decided by the unit vector and by a predefined magnitude the known bond length from O5'-P
        else:
            p_bead = traj.xyz[:, atom_map["P"], :] # for all other residues, use the P atom

        # collecting backbone
        o5_bead = traj.xyz[:, atom_map["O5'"], :] # extract XYZ coordinate of the O5'
        c5_bead = traj.xyz[:, atom_map["C5'"], :] # extract XYZ coordinate of the C5'
        c4_bead = traj.xyz[:, atom_map["C4'"], :] # extract XYZ coordinate of the C4'
        c1_bead = traj.xyz[:, atom_map["C1'"], :] # extract XYZ coordinate of the C1'
        backbone = [p_bead, o5_bead, c5_bead, c4_bead, c1_bead] # build backbone list of 5 arrays, each (n_frames, 3), bead indices are fixed here (indices 0-4) which allows for predefined quadruplets 

        # Base beads calculation
        base_beads = [] # list to store base beads
        for key, atom_list in CG_bases.items(): # loop over CG bases dictionary 
            """
            Explanation of variables invovled because it hurts my head to think about this

            key - 	A1, A2, C1, etc. Predefined in the CG bases dictionary
            atom list - ["N7", "N9", "C4", "C5", "C8"], etc. Predefined and associated with a specific base in the CG_bases dictionary
            resname - full residue name in topology defined by MDTraj (ADE, GUA, etc)
            resname[0] - single letter residue code (A, C, T, etc.)

            Edge case - missing atoms
                If an atom listed in CG_bases is absent in the topology (e.g. incomplete residue), the "if atom_name in atom_map" line silently skips it.
                This prevents crashes but could affect the COM location.
            """
            if key.startswith(resname[0]): # match the current nucleotide with one in the dictionary (A,G,C,T,U). By checking the first letter we automatically pick up both purine keys (A1, A2), but only 1 key for pyrimidines (C1, U1, T1)
                # need all atom coordinates in each frame and their element types to calculate mass-weighted centre-of-mass.
                coords_list = [] # list of arrays, each (n_frames, 3 coordinates) for each atom
                elements_list = [] # list to store atom elements, aligned with each coordinate list in coords_list
                for atom_name in atom_list: # loop over every predefined atom for the base
                    if atom_name in atom_map: # match the atom name in the CG bases dictionary with the atom in the constructed atom dictionary for this residue
                        coords_list.append(traj.xyz[:, atom_map[atom_name], :]) # extract XYZ coordinates and append to list
                        elements_list.append(top.atom(atom_map[atom_name]).element.symbol) # extract atom element and append to list
                if coords_list: # only continue if at least 1 atom has been extracted
                    coords_arr = np.stack(coords_list, axis=1) # aligns all selected atoms side-by-side for a given frame so we can treat each frame independently
                    coms = np.array([compute_com(coords_arr[i], elements_list, atomic_weights) for i in range(n_frames)]) # loop over frames so each COM is computed from atoms within the same snapshot, giving a time series of base positions
                    base_beads.append(coms) # append centre of mass to list
                    # For a pyrimidine, base_beads ends up with one element: (n_frames, 3 coordinates) 
                    # For a purine, the loop matches twice (A1 and A2), so base_beads becomes:(n_frames, 3 coordinates), (n_frames, 3 coordinates) 

        # Combine all beads for residue
        if base_beads:
            full_beads = backbone + [bead[:, np.newaxis, :] for bead in base_beads]
        else:
            full_beads = backbone # if no base beads made then just use rest of backbone
            
        reshaped_beads = [b if b.ndim == 3 else b[:, np.newaxis, :] for b in full_beads] # check that all beads are in shape (n_frames, 1, 3)
        residue_beads = np.concatenate(reshaped_beads, axis=1)  # concatenate into a full (n_frames, n_beads, 3 coordinates) array
        # Final residue_beads shape
            # Purine: (n_frames, 7 beads, 3 coordinates)
            # Pyrimidine: (n_frames, 6 beads, 3 coordinates)

        # Select quadruplets based on residue type
        if resname[0] in ['A', 'G']:  # check if start of residue matches with shorthand purine base symbols
            dihedrals = compute_dihedrals(residue_beads, PURINE_QUADRUPLETS) # compute dihedrals using predefined function
        else:  # if not then is a pyrimidine
            dihedrals = compute_dihedrals(residue_beads, PYRIMIDINE_QUADRUPLETS)
            
        dihedral_trajs.append(dihedrals) # append to list 

    # Combine dihedrals across residues
    dihedrals_combined = np.concatenate(dihedral_trajs, axis=1)  # once all dihedrals calculated concatenate all residue dihedrals into one array (n_frames, n_dihedrals_total)
    
    # Extract start, end, and path
    start = dihedrals_combined[0]
    end = dihedrals_combined[-1]
    path = dihedrals_combined
    
    all_paths.append((start, end, path))

print(f"Processed {len(all_paths)} trajectories. Dihedrals per frame: {all_paths[0][2].shape[1]}") # debugging line, check how many samples were processed
# True  → at least one NaN present anywhere in the dataset
# False → dataset is completely NaN‑free
print("Any NaNs? →", any(np.isnan(x).any() for triplet in all_paths for x in triplet)) # debugging line, check if any NANs were introduced (e.g. due to bad input or division by zero in dihedral calculation)

# check that the dihedrals are consistent across the data samples
n_dihedrals = all_paths[0][2].shape[1]  # get number of dihedrals from the first sample's path
assert all(path.shape[1] == n_dihedrals for (_, _, path) in all_paths), \
       "Inconsistent dihedral counts!"

from Landscape_DDPM_v2 import normalize_dihedrals
normalized_paths = []
for start, end, path in all_paths:
    norm_start = normalize_dihedrals(start)  # (n_dihedrals,)
    norm_end = normalize_dihedrals(end)      # (n_dihedrals,)
    norm_path = normalize_dihedrals(path)    # (n_frames, n_dihedrals)
    normalized_paths.append((norm_start, norm_end, norm_path))

# While training a model, we typically want to pass samples in batches and reshuffle the data at every epoch to reduce model overfitting
# DataLoader is an iterable that abstracts this complexity in an easy API.
from Landscape_DDPM_v2 import MolecularPathDataset, collate_paths
from torch.utils.data import DataLoader

dataset = MolecularPathDataset(normalized_paths) # use all_paths not all atom coordinates
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_paths)

from Landscape_DDPM_v2 import GaussianDiffusion, Trainer, UNet

# Get dihedral dimensions from the first sample
sample = next(iter(dataloader))
n_dihedrals = sample['path'].shape[-1]  # Shape is (B, T, n_dihedrals)

model = UNet(
    input_dim=n_dihedrals,  
    base_dim=32,
    dim_mults=(1, 2, 2, 4),
    time_emb_dim=128,
    out_dim=None         # same as in_dim by default
)


diffusion_model = GaussianDiffusion(
    model,                        # U-net model
    timesteps = 1000,             # number of diffusion steps 
    loss_type='periodic'
)

trainer = Trainer(
    diffusion=diffusion_model,           # Your GaussianDiffusion instance
    dataloader=dataloader,               # From your earlier collate_paths function
    ema_decay=0.995,
    learning_rate=1e-3,
    results_folder='.../Models',
    save_name='molecular_path_diffusion_test.pt',
    device='cuda',
    use_amp=False
)
trainer.train(num_epochs=100)

import torch
from Landscape_DDPM_v2 import GaussianDiffusion, UNet

# Get dihedral dimensions from the first sample
sample = next(iter(dataloader))
n_dihedrals = sample['path'].shape[-1]  # Shape is (B, T, n_dihedrals)

model = UNet(
    input_dim=n_dihedrals,           # atoms × 3 coordinates
    base_dim=32,
    dim_mults=(1, 2, 2, 4),
    time_emb_dim=128,
    out_dim=None         # same as in_dim by default
)

checkpoint = torch.load(".../Models/molecular_path_diffusion_test.pt")
model.load_state_dict(checkpoint['ema'])
# model = model.to('cuda') # move model to cuda, hardcoded fix
model.eval() # set to evaluate mode so weights don't change

# Load diffusion wrapper
diffusion = GaussianDiffusion(model, timesteps=1000)

# Configuration
PO5_BOND_LENGTH = 0.1615  # nm

# Atomic weights for COM calculations
atomic_weights = {"C": 12.011, "N": 14.007, "O": 16.000}

# Nucleotide base definitions
CG_bases = {
    "A1": ["N7", "N9", "C4", "C5", "C8"],
    "A2": ["N1", "C2", "N3", "C4", "C5", "N6", "C6"],
    "C1": ["N1", "C2", "N3", "N4", "C4", "C5", "C6", "O2"],
    "G1": ["N7", "N9", "C4", "C5", "C8"],
    "G2": ["N1", "N2", "C2", "N3", "C4", "C5", "C6", "O6"],
    "T1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"],
    "U1": ["N1", "C2", "N3", "C4", "C5", "C6", "O2", "O4"]
}

# Quadruplet definitions for dihedrals (0-based indexing)
PURINE_QUADRUPLETS = [  # For A, G (10 dihedrals)
    (3, 4, 5, 6),   # 0: R4-R1-A1/G1-A2/G2
    (3, 5, 6, 4),   # 1: R4-A1/G1-A2/G2-R1
    (2, 3, 4, 5),   # 2: C-R4-R1-X1
    (0, 3, 4, 5),   # 3: P-R4-R1-X1
    (2, 3, 0, 1),   # 4: C-R4-P-O
    (4, 3, 0, 1),   # 5: R1-R4-P-O
    (1, 2, 3, 0),   # 6: O-C-R4-P
    (1, 2, 3, 4),   # 7: O-C-R4-R1
    (0, 1, 2, 3),   # 8: P-O-C-R4
    (3, 0, 1, 2)    # 9: R4-P-O-C
]

PYRIMIDINE_QUADRUPLETS = [  # For C, T, U (8 dihedrals, skipping 0 and 1)
    (2, 3, 4, 5),   # 2: C-R4-R1-X1 
    (0, 3, 4, 5),   # 3: P-R4-R1-X1 
    (2, 3, 0, 1),   # 4: C-R4-P-O 
    (4, 3, 0, 1),   # 5: R1-R4-P-O 
    (1, 2, 3, 0),   # 6: O-C-R4-P 
    (1, 2, 3, 4),   # 7: O-C-R4-R1 
    (0, 1, 2, 3),   # 8: P-O-C-R4 
    (3, 0, 1, 2)    # 9: R4-P-O-C 
]

# process a single path to condition the inference
traj = md.load_mdcrd('.../path.mdcrd', top=prmtop_file)
top = traj.topology
n_frames = traj.n_frames

# just copy paste the previouse dihedral calculating code

dihedral_trajs = []  # Store dihedrals per residue per frame

for res in top.residues: # loop over each residue
        atom_map = {a.name: a.index for a in res.atoms} # create a dictionary of atom names to atom index in the residue
        resname = res.name.strip() # get residue name

        # backbone bead calculation
        if res.index == 0: # for the 1st residue 
            ho5 = traj.xyz[:, atom_map["HO5'"], :] # extract XYZ coordinate of the HO5
            o5 = traj.xyz[:, atom_map["O5'"], :] # extract XYZ coordinate of the O5'
            direction = o5 - ho5 # find a direction vector between these atoms, should be the bond HO5'-O5'
            unit_vec = direction / (np.linalg.norm(direction, axis=1, keepdims=True) + 1e-8) # normalize the bond vector for each frame to obtain a direction-only unit vector
                                                                                             # axis=1 computes norm along x/y/z dimensions
                                                                                             # + 1e-8 avoids division-by-zero
            p_bead = o5 + unit_vec * PO5_BOND_LENGTH # from the O5' atom scale the bond length in a direction decided by the unit vector and by a predefined magnitude the known bond length from O5'-P
        else:
            p_bead = traj.xyz[:, atom_map["P"], :] # for all other residues, use the P atom

        # collecting backbone
        o5_bead = traj.xyz[:, atom_map["O5'"], :] # extract XYZ coordinate of the O5'
        c5_bead = traj.xyz[:, atom_map["C5'"], :] # extract XYZ coordinate of the C5'
        c4_bead = traj.xyz[:, atom_map["C4'"], :] # extract XYZ coordinate of the C4'
        c1_bead = traj.xyz[:, atom_map["C1'"], :] # extract XYZ coordinate of the C1'
        backbone = [p_bead, o5_bead, c5_bead, c4_bead, c1_bead] # build backbone list of 5 arrays, each (n_frames, 3), bead indices are fixed here (indices 0-4) which allows for predefined quadruplets 

        # Base beads calculation
        base_beads = [] # list to store base beads
        for key, atom_list in CG_bases.items(): # loop over CG bases dictionary 
            """
            Explanation of variables invovled because it hurts my head to think about this

            key - 	A1, A2, C1, etc. Predefined in the CG bases dictionary
            atom list - ["N7", "N9", "C4", "C5", "C8"], etc. Predefined and associated with a specific base in the CG_bases dictionary
            resname - full residue name in topology defined by MDTraj (ADE, GUA, etc)
            resname[0] - single letter residue code (A, C, T, etc.)

            Edge case - missing atoms
                If an atom listed in CG_bases is absent in the topology (e.g. incomplete residue), the "if atom_name in atom_map" line silently skips it.
                This prevents crashes but could affect the COM location.
            """
            if key.startswith(resname[0]): # match the current nucleotide with one in the dictionary (A,G,C,T,U). By checking the first letter we automatically pick up both purine keys (A1, A2), but only 1 key for pyrimidines (C1, U1, T1)
                # need all atom coordinates in each frame and their element types to calculate mass-weighted centre-of-mass.
                coords_list = [] # list of arrays, each (n_frames, 3 coordinates) for each atom
                elements_list = [] # list to store atom elements, aligned with each coordinate list in coords_list
                for atom_name in atom_list: # loop over every predefined atom for the base
                    if atom_name in atom_map: # match the atom name in the CG bases dictionary with the atom in the constructed atom dictionary for this residue
                        coords_list.append(traj.xyz[:, atom_map[atom_name], :]) # extract XYZ coordinates and append to list
                        elements_list.append(top.atom(atom_map[atom_name]).element.symbol) # extract atom element and append to list
                if coords_list: # only continue if at least 1 atom has been extracted
                    coords_arr = np.stack(coords_list, axis=1) # aligns all selected atoms side-by-side for a given frame so we can treat each frame independently
                    coms = np.array([compute_com(coords_arr[i], elements_list, atomic_weights) for i in range(n_frames)]) # loop over frames so each COM is computed from atoms within the same snapshot, giving a time series of base positions
                    base_beads.append(coms) # append centre of mass to list
                    # For a pyrimidine, base_beads ends up with one element: (n_frames, 3 coordinates) 
                    # For a purine, the loop matches twice (A1 and A2), so base_beads becomes:(n_frames, 3 coordinates), (n_frames, 3 coordinates) 

        # Combine all beads for residue
        if base_beads:
            full_beads = backbone + [bead[:, np.newaxis, :] for bead in base_beads]
        else:
            full_beads = backbone # if no base beads made then just use rest of backbone
            
        reshaped_beads = [b if b.ndim == 3 else b[:, np.newaxis, :] for b in full_beads] # check that all beads are in shape (n_frames, 1, 3)
        residue_beads = np.concatenate(reshaped_beads, axis=1)  # concatenate into a full (n_frames, n_beads, 3 coordinates) array
        # Final residue_beads shape
            # Purine: (n_frames, 7 beads, 3 coordinates)
            # Pyrimidine: (n_frames, 6 beads, 3 coordinates)

        # Select quadruplets based on residue type
        if resname[0] in ['A', 'G']:  # check if start of residue matches with shorthand purine base symbols
            dihedrals = compute_dihedrals(residue_beads, PURINE_QUADRUPLETS) # compute dihedrals using predefined function
        else:  # if not then is a pyrimidine
            dihedrals = compute_dihedrals(residue_beads, PYRIMIDINE_QUADRUPLETS)
            
        dihedral_trajs.append(dihedrals) # append to list 

# Combine dihedrals across residues
dihedrals_combined = np.concatenate(dihedral_trajs, axis=1)  # (n_frames, 10 * n_residues)

# Extract start and end structure and convert to tensors for inference
start = torch.tensor(dihedrals_combined[0] / np.pi, dtype=torch.float32).unsqueeze(0)
end = torch.tensor(dihedrals_combined[-1] / np.pi, dtype=torch.float32).unsqueeze(0)
# run inference 
generated_path_norm = diffusion.sample(model, start, end, frames=411, device='cuda')  # (1, 50, F)
generated_path = generated_path_norm * np.pi # convert to radians

from Landscape_DDPM_v2 import reconstruct_hire_cg

# configuration
atomic_mass = {"C":12.011,"N":14.007,"O":16.000}

# bead order used by the reconstructor, defines the canonical bead order used 
# first 5 beads (P, O5, C5, C4, C1) make up the common backbone
# Purines (A, G) have two base beads: X1, X2
# Pyrimidines (C, U, T) only have X1
BEAD_ORDER = {
    "A": ["P", "O5", "C5", "C4", "C1", "A1", "A2"],
    "G": ["P", "O5", "C5", "C4", "C1", "G1", "G2"],
    "C": ["P", "O5", "C5", "C4", "C1", "C1"],
    "U": ["P", "O5", "C5", "C4", "C1", "U1"],
    "T": ["P", "O5", "C5", "C4", "C1", "T1"]
}

# backbone bond length table 

BL_CONST_A = { #  (res, bead_1, bead_2) : length in angstroms
    # * is a wildcard: it applies to all nucleotides
    ("*", "C4", "P")     : 3.800,
    ("*", "C4", "C1")    : 2.344,
    ("A", "C1", "A1")    : 2.633,
    ("G", "C1", "G1")    : 2.622,
    ("U", "C1", "U1")    : 3.062,
    ("C", "C1", "C1")    : 3.004,  
    ("G", "G1", "G2")    : 2.450,
    ("A", "A1", "A2")    : 2.180,
    ("*", "C5", "C4")    : 1.520,
    ("*", "P",  "O5")    : 1.593,
    ("*", "O5", "C5")    : 1.430
}

# backbone bond angle table in degrees
# R4, R1, O are aliases for canonical beads like C4, C1, O5
# Some are specific to nucleotide types (e.g., (A, R4, R1, A1)) while others apply universally
BA_CONST_DEG = {
    ("A", "C4", "C1", "A1") : 123.6,
    ("U", "C4", "C1", "U1") : 132.5,
    ("G", "C4", "C1", "G1") : 123.5,
    ("C", "C4", "C1", "C1") : 131.1,  
    ("A", "C1", "A1", "A2") : 116.7,
    ("G", "C1", "G1", "G2") : 111.2,
    ("*", "P",  "O5", "C5") : 122.9,
    ("*", "O5", "C5", "C4") : 110.6,
    ("*", "C5", "C4", "P")  :  98.0,
    ("*", "C4", "P",  "O5") : 110.0,
    ("*", "C5", "C4", "C1") : 135.5,
    ("*", "C1", "C4", "P")  :  98.0
}

# mapping of aliases to canonical bead names
ALIASES = {"O":"O5","R4":"C4","R1":"C1"}

from Landscape_DDPM_v2 import geometry_parameters

traj = md.load_mdcrd('...path.mdcrd', top=prmtop_file)
top = traj.topology

assert not np.isnan(generated_path.cpu().numpy()).any(), "NaNs found in sampled dihedrals!" # check for issues with inference
dihedrals_in_radians = generated_path.squeeze(0).cpu().numpy()
residue_types = [res.name.strip()[0] for res in top.residues]

bead_order, bond_lengths, bond_angles = geometry_parameters(
        trajectory_folder=trajectory_folder,    
        prmtop_file=prmtop_file,
        bead_orders=BEAD_ORDER, 
        bond_length_constants=BL_CONST_A,
        bond_angles_constants=BA_CONST_DEG,
        atomic_weights= atomic_mass
)

# After you have (dihedrals, residue_types, bead_order, bond_length, bond_angle):
print('starting reconstruction')
coords = reconstruct_hire_cg(
            dihedrals      = dihedrals_in_radians,
            residue_types  = residue_types,          # e.g. ["G","A","C"]
            bead_orders    = bead_order,           # from extractor
            bond_lengths   = bond_lengths,      # from extractor
            bond_angles    = bond_angles        # from extractor
         )

from Landscape_DDPM_v2 import save_reconstructed_xyz

print('saving XYZ file')
save_reconstructed_xyz(
    coordinates=coords,
    filename="generated_path_dihedral.xyz",
    residue_types=residue_types  # e.g. ['A', 'A', 'G', 'U', ...]
)
