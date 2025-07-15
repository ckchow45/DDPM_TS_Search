# GenAI_TS_Search
Repository of code for my MSC Applied Bioinformatics research project on developing a generative Denoising Diffusion Probabilistic Model (DDPM) to help transition state searches in discrete path sampling

## All-Atom Coordinate Model
This branch contains the code and training scripts required to train the all-atom coordinate DDPM and generate an all-atom coordinate path.

Assuming you have data generated from the path_generation.py file and converted them into .mdcrd format you can run the scripts to train your own model.

The training script has been separated from the inference script. Make sure to run training_script.py before running inference_script.py

