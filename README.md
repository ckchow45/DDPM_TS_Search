# GenAI_TS_Search
Repository of code for my MSC Applied Bioinformatics research project on making a generative Denoising Diffusion Probabilistic Model (DDPM) for transition state searches in discrete path sampling

## The 3 models
3 different models were made, with each kept on separate branches

  -The All-atom coordinate model is on the All_atom_model branch

  -The CG model is on the CG_model branch

  -The dihedral model is on the Dihedral_model branch

Each model will have a script containing all the functions and classes that are called, and a separate training script that handles the data processing pipeline and model training
Please note that all the file paths will need to be updated if you wish to run these as they have been replaced with generic names

## References
The DDPM model is largely based on the model made by Wang, Heron and Tiwary, with some modifications to allow for conditioning. Their paper can be found here:

Wang Y, Herron L, Tiwary P. From data to noise to data for mixing physics across temperatures with generative artificial intelligence. Proc Natl Acad Sci U S A [Internet]. 2022 Aug 9 [cited 2025 May 30];119(32):e2203656119. Available from: https://www.pnas.org/doi/abs/10.1073/pnas.2203656119
