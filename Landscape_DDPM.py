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

def periodic_mse_loss(pred, target):
    """
    Periodic-aware MSE loss for dihedrals.
    Args:
        pred: Model predictions in [-1, 1] (normalized)
        target: Ground truth in [-1, 1] (normalized)
    Returns:
        MSE loss accounting for periodicity.
    """
    # Convert to radians temporarily
    pred_rad = pred * torch.pi
    target_rad = target * torch.pi
    
    # Smallest angular difference
    diff = torch.atan2(torch.sin(pred_rad - target_rad), 
                      torch.cos(pred_rad - target_rad))  # [-π, π]
    
    # Convert back to [-1, 1] range for stable training
    diff_normalized = diff / torch.pi
    return (diff_normalized ** 2)

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
def compute_com(coords, elements):
    """
    Compute center of mass from coordinates and atom types.
        - coords = A NumPy array of shape (n_atoms, 3 coordinates), each row is the 3D coordinate [x, y, z] of one atom
        - elements = A list of N atomic element symbols in the same order as the coordinates
    """
    weights = np.array([atomic_weights[e] for e in elements]) # For each atom element, look up its atomic weights from the predefined dictionary, outputs a vector of atomic masses
    return np.average(coords, axis=0, weights=weights) # calculate masses using a weighted average (xyz coords are averaged using the atomic weights as weights)

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
        # beta scheduler is the method used to define how much noise (βₜ) we add at each timestep
        # βₜ (beta at timestep t) is the variance of the Gaussian noise added at that step, controlling how much noise is added
        # used because:
            #The early timesteps are clearer — data is only slightly noisy.
            #The later ones make the model robust to heavier noise.
        if beta_schedule == 'linear': # currently only linear beta scheduler is implemented
            self.betas = torch.linspace(1e-4, 0.02, timesteps) # creates a tensor of timesteps values evenly spaced from 1e-4 to 0.02, beta starts small to larger values
        else:
            raise NotImplementedError(f"Beta schedule '{beta_schedule}' not supported.") # give error if any other scheduler is given

        # Precompute alpha and alpha_bar values
        # for all timesteps compute:
            #αₜ = 1 - βₜ (what portion of the original data (signal) survives at each step)
            #ᾱₜ = ∏ₛ₌₀^ₜ αₛ (how much of the original signal survives up to step t)
        # this is done as wiithout knowing how much is left (ᾱₜ), you can't reconstruct how it got there or how to bring it back, also means no need to simulate all previous timesteps
        self.alphas = 1. - self.betas # αₜ = 1 - βₜ for each timestep — the retention factor for signal at each step.
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # Computes ᾱₜ = ∏ₛ₌₀^ₜ αₛ — the cumulative product of α’s up to timestep t.
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod) # reparameterize noisy inputs during training (for efficient sampling of xₜ from x₀ + noise).
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod) # reparameterize noisy inputs during training (for efficient sampling of xₜ from x₀ + noise).

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
        At each timestep t, we:
            Predict the noise
            μθ​(xt​,t): the expected denoised sample, based on current noisy input xt​ and time step t.
            Use DDPM denoising formula to get a mean μθμθ​

            Optionally add new noise scaled

            Repeat for t - 1, until t = 0

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

            # Retrieve the precomputed scalar coefficients needed for the reverse diffusion equation at timestep t
            # These control the noise schedule and how much signal vs. noise is retained or added at each timestep.
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_bar_t = self.alphas_cumprod[t]
            alpha_bar_prev = self.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0).to(device)
            # Compute constants used in the denoising update step
            # analytically compute how to go backward from xt→xt−1xt​→xt−1​ without needing to simulate each step again.
            coef1 = 1 / torch.sqrt(alpha_t)
            coef2 = (1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)

            # Updates x with the predicted noise
            x = coef1 * (x - coef2 * pred_noise)

            # Adds back Gaussian noise (until the final step (t = 0), where we want a clean sample) scaled by σ_t to maintain the correct level of randomness in the trajectory.
            if t > 0:
                noise = torch.randn_like(x)
                sigma_t = torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar_t) * beta_t)
                x += sigma_t * noise

        return x * torch.pi

    def forward(self, x, model, mask=None, start=None, end=None): # the training pass for the diffusion model.
        """
        Training loss computation.

        Args:
            x: (B, T, F) — path
            model: the UNet model
            mask: (B, T) — binary mask (1 for valid steps, 0 for padding)
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
            loss = periodic_mse_loss(pred_noise, noise)
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

    def forward(self, t): # forward pass, t is a tensor of shape (B,) — one timestep per batch element (usually int64 or float32).
        device = t.device # move to selected device
        half_dim = self.dim // 2 # split the full dimension in half — because we compute sine for the first half, and cosine for the second half.
        emb = torch.exp(torch.arange(half_dim, device=device) * -(torch.log(torch.tensor(10000.0)) / half_dim)) # Create a tensor of size half_dim with exponentially scaled values to control sine/cosine frequency 
        emb = t.unsqueeze(-1) * emb # turns (B,) into (B, 1) so it can be multiplied element-wise with the emb vector, giving tensor of shape (B, half_dim)
        # Each row is now a vector of scaled time values for sine and cosine functions.
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) # take the sine of the first half and cosine of the second half and concatenate the two halves
        return emb # results in a smooth, continuous embedding of the timestep that carries both high-frequency and low-frequency information — helping the model know “how far along” the diffusion process is.

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
    # pre activation (Normalization → Activation → Convolution) is used over post activation (Convolution → Normalization → Activation) as a paper 'https://arxiv.org/pdf/1603.05027' shows that this is slightly better for ResNet
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
        inner_dim = dim_head * heads # Total inner dimension for query/key/value is dim_head × heads
        self.to_qkv = nn.Conv1d(dim, inner_dim * 3, 1, bias=False) # Applies a 1D convolution to project input x into Q, K, and V, each of shape (inner_dim, t)
                                                                   # Since we need 3 sets (Q, K, V), the output has size 3 × inner_dim.
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

        out = out.reshape(b, -1, t) # Reshape from (batch, heads, dim_head, time) → (batch, inner_dim, time)
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
        t_emb = self.time_mlp(t[:, 0])  # Apply time embedding MLP to the first timestep of each batch sample (B,) → (B, time_emb_dim)

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
        return x.permute(0, 2, 1)  # (B, F, T) → (B, T, F)
