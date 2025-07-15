# GenAI_TS_Search
This branch contains the codebase for training and running the dihedral angle–based DDPM model, which generates coarse-grained RNA backbone structures using predicted dihedrals and fixed internal bond information. These predicted dihedrals are converted into full 3D coordinates through reconstruction with the aforementiond internal coordinate information

## Dependencies
Ensure that you have the following:
- numpy, torch, tqdm, mdtraj, matplotlib (for RMSD/trajectory evaluation)
- The dataset in `.mdcrd` format
- A valid `prmtop` file for topology

## Dihedral Model
This branch contains the code and training scripts required to train the dihedral DDPM and generate dihedrals which can processed into a CG coordinate path.

The model operates on dihedral angles rather than Cartesian or coarse-grained bead coordinates. It is the most successful model among the three in generating valid RNA-like structures, although intermediate frame quality still varies

The training code and the inference code are kept on the same script so you can just run dihedral_training.py once

All the background code on the actual UNet, beta schedule and general model architecture as well as the reconstruction code that is imported into the script can be found in Landscape_DDPM_v2.py

## Usage
Assuming you have data generated from the `path_generation.py` file and converted them into `.mdcrd` format you can run the `dihedral_training.py` script to train your own model and generate a path

Make sure to update the file paths in the script to match your local directory structure

Optionally, you can use the `RMSD_graph.ipynb` notebook on the main branch to evaluate the generated path with RMSD to start and end

## References
The DDPM model is largely based on the model made by Wang, Heron and Tiwary, with some modifications to allow for conditioning. Their paper can be found here:

Wang Y, Herron L, Tiwary P. From data to noise to data for mixing physics across temperatures with generative artificial intelligence. Proc Natl Acad Sci U S A [Internet]. 2022 Aug 9 [cited 2025 May 30];119(32):e2203656119. Available from: https://www.pnas.org/doi/abs/10.1073/pnas.2203656119
