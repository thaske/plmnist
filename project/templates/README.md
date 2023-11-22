## templates directory
----------------------

### Overview
This directory is used to store the custom HPC submission scripts and any template files used for the custom workflow (Example: A base template file that is used for a find and replace function, changing the variables with the differing state point inputs).  These find and replace template files could also be put in the `src` directory, but the HPC submission scripts must remain in the `templates` directory.   **All the standard or custom module load commands, conda activate commands, and any other custom items that needed to be HPC submission scripts should in included here for every project (Example: Specific queues, CPU/GPU models, etc.).** 

This specific `phoenix.sh` file is designed to auto-select and submit CPU or GPU Slurm scripts based on what CPUs only or GPUs the user selected for each part of the project (i.e., jobs in the project parts.)