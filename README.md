# GenAI_TS_Search
Repository of code for my MSC Applied Bioinformatics research project on developing a generative Denoising Diffusion Probabilistic Model (DDPM) to help transition state searches in discrete path sampling

## All-Atom Coordinate Model
This branch contains the code and training scripts required to train the all-atom coordinate DDPM and generate an all-atom coordinate path.

Assuming you have data generated from the path_generation.py file and converted them into .mdcrd format you can run the scripts to train your own model.

The training script has been separated from the inference script. Make sure to run training_script.py before running inference_script.py

All the background code on the actual UNet, beta schedule and general model architecture that is imported into the scripts can be found in Landscape_DDPM.py

## References
The DDPM model is largely based on the model made by Wang, Heron and Tiwary, with some modifications to allow for conditioning. Their paper can be found here:

Wang Y, Herron L, Tiwary P. From data to noise to data for mixing physics across temperatures with generative artificial intelligence. Proc Natl Acad Sci U S A [Internet]. 2022 Aug 9 [cited 2025 May 30];119(32):e2203656119. Available from: https://www.pnas.org/doi/abs/10.1073/pnas.2203656119
