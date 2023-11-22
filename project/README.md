## Project directory
--------------------

### Overview
All the `signac` commands are run from the `<local_path>/signac_pytorch_plmnist_example/signac_pytorch_plmnist_example/project` directory, which include, but are not limited to the following commands:
 - State point initialization.
```bash
python init.py
```
 - Checking the project status.
```bash
python project.py status
```
 - Running the project's selected part locally (general example only).
```bash
python project.py run -o <part_x_this_does_a_function_y>
```

  - Submitting the project's selected part to the HPC (general example only).
```bash
python project.py submit -o <part_x_this_does_a_function_y>
```

When using the `run` command, you can run the following flags, controlling the how the jobs are executed on the local machine (does not produce HPC job submission scripts):
 - `--parallel 2` : This only works this way when using `run`. This runs several jobs in parallel (2 in this case) at a time on the local machine, auto adjusting the time, CPU cores, etc., based on the total command selections.
 - See the `signac` [documenation](https://docs.signac.io/en/latest/) for more information, features, and the [Project Command Line Interface](https://docs.signac.io/projects/flow/en/latest/project-cli.html).

When using the `submit` command, you can run the following flags, controlling the how the jobs are submitted to the HPC:
 - `--bundle 2` : Only available when using `submit`.  This bundles multiple jobs (2 in this case) into a single run or HPC submittion script, auto adjusting the time, CPU cores, etc., based on the total command selections.
  - `--pretend` : Only available when using `submit`.  This is used to output what the submission script will look like, without submitting it to the HPC. 
  - `--parallel` : This only works this way when using `submit`.  The `N` value in `--parallel N` is not readl; therefore, it only runs all the jobs in a HPC submittion script at the same time (in parallel), auto adjusting the time, CPU cores, etc., based on the total command selections. 
  - See the `signac` [documenation](https://docs.signac.io/en/latest/) for more information, features, and the [Project Command Line Interface](https://docs.signac.io/projects/flow/en/latest/project-cli.html).


 ## templates directory and hpc_setup.py file
---------------------------------------------

### hpc_setup.py file
This file, `hpc_setup.py`, is used to specify the the HPC environment.  The `class` will need to be setup for each HPC (changing the class name and Default environment).  The following also need changed in the `class`:
 - The `template` variable to changed to the custom HPC submission script (the `slurm` `phoenix.sh` file is used here), which is located in the `templates` directory.  
 - The `hostname_pattern` variable to changed to the custom HPC hostname. In this case 'hostname' produced 'login-phoenix-slurm-2.pace.gatech.edu'.  

### Templates directory
This directory is used to store the custom HPC submission scripts and any template files used for the custom workflow (Example: A base template file that is used for a find and replace function, changing the variables with the differing state point inputs).  These find and replace template files could also be put in the `src` directory, but the HPC submission scripts must remain in the `templates` directory.   **All the standard or custom module load commands, conda activate commands, and any other custom items that needed to be HPC submission scripts should in included here for every project (Example: Specific queues, CPU/GPU models, etc.).** 

This specific `phoenix.sh` file is designed to auto-select and submit CPU or GPU Slurm scripts based on what CPUs only or GPUs the user selected for each part of the project (i.e., jobs in the project parts.)