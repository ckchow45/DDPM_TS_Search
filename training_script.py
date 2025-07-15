import mdtraj as md
import numpy as np
import os

# Configuration
trajectory_folder = '/.../Paths'       # folder containing all .mdcrd files
topology_file = '.../coords.prmtop'  # shared topology

all_paths = []

for file in os.listdir(trajectory_folder): #loop over all mdcrd files in the directory
    if file.endswith(".mdcrd"): #check if the file ends with .mdcrd to make sure that only mdcrd files are selected
        filepath = os.path.join(trajectory_folder, file) #get the full filepath of the path file
        print(f"Processing: {file}")
        
        traj = md.load_mdcrd(filepath, top=topology_file) #load the trajectory into python

        coords = traj.xyz #get the xyz (Cartersian) coordinates of the trajectories as a numpy array
                          #all of the distances in the Trajectory are stored in nanometers. The time unit is picoseconds. Angles are stored in degrees (not radians).
        
        flattened = coords.reshape(coords.shape[0], -1) #need to flatten the frames into 1D for diffusion U-net model to accept
        #goal is to predict paths between 2 endpoints so training samples should have a start and end point as well
        #create tuples like (start_frame, end_frame, path) for training data
        start = flattened[0]    # shape: (n_atoms * 3,), get the 1st frame of the path
        end = flattened[-1]    # shape: (n_atoms * 3,), get the last frame of the path
        path = flattened # the entire path
        all_paths.append((start, end, path)) #append the tuple to a new list

# While training a model, we typically want to pass samples in batches and reshuffle the data at every epoch to reduce model overfitting
# DataLoader is an iterable that abstracts this complexity in an easy API from pytorch
from Landscape_DDPM import MolecularPathDataset, collate_paths
from torch.utils.data import DataLoader

dataset = MolecularPathDataset(all_paths) # shove into a custom pytorch dataset class
dataloader = DataLoader(dataset, batch_size=50, shuffle=True, collate_fn=collate_paths) # insert into dataloader

from Landscape_DDPM import GaussianDiffusion, Trainer, UNet

# Model setup
# calculate input dimensions
n_atoms = 708
F = n_atoms * 3

# define the U-net structure
model = UNet(
    input_dim=F,           # atoms × 3 coordinates
    base_dim=32,
    dim_mults=(1, 2, 2, 4),
    time_emb_dim=128,
    out_dim=None         # same as in_dim by default
)

# wrap the Unet in GaussianDiffusion wrapper to handle noise scheduling
diffusion_model = GaussianDiffusion(
    model,                        # Unet model
    timesteps = 1000,             # number of diffusion steps 
)

# Setup training loop and hyperparameters
trainer = Trainer(
    diffusion=diffusion_model,           # Diffusion model
    dataloader=dataloader,               # data input contained in a dataloader from pytorch
    ema_decay=0.995,
    learning_rate=1e-3,
    results_folder='.../Models',
    save_name='molecular_path_diffusion_attention.pt',
    device='cuda'
)

# Train
trainer.train(num_epochs=100)
