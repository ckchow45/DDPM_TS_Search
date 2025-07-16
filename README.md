# GenAI_TS_Search
This branch contains the codebase for training and running the all-atom coordinate DDPM model, which generates full discrete paths of full atom structures.

## Dependencies
Ensure that you have the following:
- numpy, torch, tqdm, mdtraj, matplotlib (for RMSD/trajectory evaluation)
- The dataset in `.mdcrd` format
- A valid `prmtop` file for topology

## All-atom coordinate Model
This branch contains the code and training scripts required to train the all-atom coordinate DDPM and generate all-atom coordinate discrete paths.

The model operates on Cartesian coordinates. The structures it generates are generally low quality and not physically coherent 

The training code and the inference code are kept on the separate scripts `training_script.py` and `inference_script.py`

All the background code on the actual UNet, beta schedule and general model architecture that is imported into the script can be found in `Landscape_DDPM.py`

## Usage
Assuming you have data generated from the `path_generation.py` file and converted them into `.mdcrd` format you can run the `training_script.py` script to train your own model. Then run `inference_script.py` to generate a path

Make sure to update the file paths in the script to match your local directory structure

Optionally, you can use the `RMSD_graph.ipynb` notebook on the main branch to evaluate the generated path with RMSD to start and end. Alternatively, you can visual the generated path in a visualiser software.

## References
Parts of this code are adapted from the DDPM implementation by Wang, Herron, and Tiwary (https://github.com/tiwarylab/DDPM_REMD) under the MIT license (https://opensource.org/license/mit/). The original license is included in this repository and their original paper can be found below.

Wang Y, Herron L, Tiwary P. From data to noise to data for mixing physics across temperatures with generative artificial intelligence. Proc Natl Acad Sci U S A [Internet]. 2022 Aug 9 [cited 2025 May 30];119(32):e2203656119. Available from: https://www.pnas.org/doi/abs/10.1073/pnas.2203656119
