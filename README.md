# DDPM_TS_Search
This repository contains the codebase for my MSc Applied Bioinformatics research project focused on developing a generative machine learning **Denoising Diffusion Probabilistic Model (DDPM)** model to aid **transition state searches** in **discrete path sampling**

## Requirements
Please make sure to install the following dependencies for the code to function:
- numpy
- torch
- MDTraj
- tqdm
- matplotlib (for optional RMSD evaluation)

All code in this repository was run under Python version 3.11.9
## Data Processing

The data can be found in the appendices of my dissertation, although it should be possible to use other energy landscape data with some modifications to the code.

Data was initially processed by converting the databases of minima and transition states into a graph before using Djikstra's path finding algorithm to construct discrete paths. Please see `path_generation.py` on the main branch for implementation.

## The 3 models
3 different models were made, with each kept on separate branches

| Branch            | Model Type            | Description                                 |
|-------------------|------------------------|--------------------------------------------|
| `All_atom_model`  | All-Atom Coordinate    | Trained off full Cartesian coordinates and generates full atomic structures            |
| `CG_model`        | Coarse-Grained (CG)    | Trained off CG bead representation and generates coarse-grained structures                |
| `Dihedral_model`  | Dihedral Angle         | Trained off dihedral angles and generates dihedral angles which are used to reconstruct structures    |

Each branch contains:
- A script (e.g., `Landscape_DDPM.py`) defining model architecture and various helper functions.
- A training script to load data, process it, train the model, and generate paths.

## Model Evaluation

Root Mean Square Deviaion (RMSD) to start and end are both calculated to evaluate the generated paths. Code for this can be found on the main branch as a notebook titled `RMSD_graph.py`

## References
Parts of this code are adapted from the DDPM implementation by Wang, Herron, and Tiwary (https://github.com/tiwarylab/DDPM_REMD) under the MIT license (https://opensource.org/license/mit/). The original license is included in this repository and their original paper can be found below.

Wang Y, Herron L, Tiwary P. From data to noise to data for mixing physics across temperatures with generative artificial intelligence. Proc Natl Acad Sci U S A [Internet]. 2022 Aug 9 [cited 2025 May 30];119(32):e2203656119. Available from: https://www.pnas.org/doi/abs/10.1073/pnas.2203656119
