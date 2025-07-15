import torch
from Landscape_DDPM import GaussianDiffusion, UNet, save_xyz
import mdtraj as md
import numpy as np

# Load in the model weights
# calculate input dimensions
n_atoms = 708
F = n_atoms * 3

# define the U-net structure
#make sure model parameters match the trained model
model = UNet(
    input_dim=F,           # atoms * 3 coordinates as input dimension
    base_dim=32,
    dim_mults=(1, 2, 2, 4),
    time_emb_dim=128,
    out_dim=None         # same as in_dim by default
)

checkpoint = torch.load(".../molecular_path_diffusion_attention.pt")
model.load_state_dict(checkpoint['ema']) # load in the EMA model weights into the defined model
model = model.to('cuda') # move model to cuda, hardcoded fix 
model.eval() # set to evaluate mode to weights don't change

# Load diffusion wrapper
diffusion = GaussianDiffusion(model, timesteps=100)

# Load in files
topology_file = '.../coords.prmtop'
topology = md.load_prmtop(topology_file)
traj = md.load_mdcrd('.../path.mdcrd' , top=topology_file)

# Prepare data for inference
coords = traj.xyz #get the xyz (Cartersian) coordinates of the trajectories as a numpy array
flattened = coords.reshape(coords.shape[0], -1) #need to flatten the frames into 1D for diffusion U-net model to accept

# Convert into tensors
start = torch.tensor(flattened[0]).float()
end = torch.tensor(flattened[-1]).float()
#add extra 1 as model expects batch and we only want 1 sample
start = start.unsqueeze(0)  # (1, F)
end = end.unsqueeze(0)      # (1, F)

print("start shape:", start.shape)  # Should be (1, F)
print("end shape:", end.shape)      # Should be (1, F)

#generate path
generated_path = diffusion.sample(model, start, end, frames=50, device = 'cuda')  # (1, 50, F)

# Extract list of atom symbols for XYZ file formatting
atom_elements = [atom.element.symbol for atom in topology.atoms]

# save outputs as XYZ file
from Landscape_DDPM import save_xyz
save_xyz(generated_path, 'generated_path_all_atom.xyz', n_atoms, atom_elements)
