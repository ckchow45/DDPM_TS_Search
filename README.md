# GenAI_TS_Search
This repository contains the codebase for my MSc Applied Bioinformatics research project focused on applying generative machine learning models—specifically a **Denoising Diffusion Probabilistic Model (DDPM)**—to aid **transition state (TS) searches** in **discrete path sampling (DPS)** workflows.

## Data Processing

Data was initially processed by converting the databases of minima and transition states into a graph before using Djikstra's path finding algorithm to construct discrete paths. Code for this can be found on the main branch as "path_generation.py"

## The 3 models
3 different models were made, with each kept on separate branches

| Branch            | Model Type            | Description                                 |
|-------------------|------------------------|--------------------------------------------|
| `All_atom_model`  | All-Atom Coordinate    | Uses full Cartesian coordinates            |
| `CG_model`        | Coarse-Grained (CG)    | Uses CG bead representation                |
| `Dihedral_model`  | Dihedral Angle         | Uses internal coordinates (angles only)    |


  -The All-atom coordinate model is on the All_atom_model branch

  -The CG model is on the CG_model branch

  -The dihedral model is on the Dihedral_model branch

Each model will have a script containing all the functions and classes that are called (Landscape_DDPM/py), and a separate training script that prepares the data for model training before calling the model to train and generate paths

Please note that all the file paths will need to be updated if you wish to run these as they have been replaced with generic names

## Model Evaluation

RMSD to start and end are both calculated to evaluate the generated paths. Code for this can be found on the main branch as a notebook titled "RMSD_graph.ipynb"

## References
The DDPM model is largely based on the model made by Wang, Heron and Tiwary, with some modifications to allow for conditioning. Their paper can be found here:

Wang Y, Herron L, Tiwary P. From data to noise to data for mixing physics across temperatures with generative artificial intelligence. Proc Natl Acad Sci U S A [Internet]. 2022 Aug 9 [cited 2025 May 30];119(32):e2203656119. Available from: https://www.pnas.org/doi/abs/10.1073/pnas.2203656119
