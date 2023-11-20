## Signac Workflow Tutorial: pl-mnist Example using Pytorch
-----------------------------------------------------------

### General Notes

Using `signac` workflows provide the following benefits: 

 - The `signac` workflows provide contained and totally reproducible results, since all the project steps and calculations are contained within this a single `signac` project. Although, to ensure total reproduciblity, the project should be run from a container.  Note: This involves building a container (Docker, Apptainer, Podman, etc.), using it to run the original calculations, and providing it the future parties that are trying to reproduce the exact results.

 - The `signac` workflows can simply track the progress of any project on locally or on the HPC, providing as much or a little details of the project status as the user programs into the `project.py` file. 

 - These `signac` workflows are designed to track the progress of all the project's parts or stages, only resubmitting the jobs locally or to the HPC if they are not completed or not already in the queque.  
 
 - These `signac` workflows also allow colleagues to quickly transfer their workflows to each other, and easily add new state points to a project, without the fear of rerunning the original state points.  

 - Please also see the [signac website](https://signac.io/), which outlines some of the other major features. 


### Overview

This is a `signac` Workflow example/tutorial for a simple Numpy calculation, which utilizes the following workflow steps:

 - **Part 1:** For each individual job (set of state points), this code generates the `signac_job_document.json` file from the `signac_statepoint.json` data.  The `signac_statepoint.json` only stores the set of state points or required variables for the given job.  The `signac_job_document.json` can be used to store any other variables that the user wants to store here for later use or searching. 

- **Part 2:** This writes the input values into a file that `Numpy` will use to do a calculation in `Part 3`.  There are four (4) random numbers generated that used the initial `value_0_int` value and the `replicate_number_int` value to seed the random number generator.

- **Part 3:** Calulate the dot product of the four (4) random numbers generated in `Part 2` (4 numbers dot [1, 2, 3, 4]).  Also, run a bash command `echo "Running the echo command or any other bash command here"`, which is an example of how to run a bash command to run a software package inside the commands for each state point. 

- **Part 4:** Obtain the average and standard deviation for each input `value_0_int` value across all the replicates, and print the output data file (`analysis/output_avg_std_of_replicates_txt_filename.txt`).  Signac is setup to automatically loop through all the json files (`signac_statepoint.json`), calculating the average and standard deviation for the jobs with the state points that only have a different `replicate_number_int` numbers. 

#### Notes:
- **src directory:** This directory can be used to store any custom function that are required for this workflow.  This includes any developed `Python` functions or any template files used for the custom workflow (Example: A base template file that is used for a find and replace function, changing the variables with the differing state point inputs).

- **templates directory:** This directory is used to store the custom HPC submission scripts and any template files used for the custom workflow (Example: A base template file that is used for a find and replace function, changing the variables with the differing state point inputs).  These find and replace template files could also be put in the `src` directory, but the HPC submission scripts must remain in the `templates` directory.  **All the standard or custom module load commands, conda activate commands, and any other custom items that needed to be HPC submission scripts should in included here for every project (Example: Specific queues, CPU/GPU models, etc.).** 

### Resources
 - The [signac documentation](https://signac.io/) and the [signac GitHub](https://github.com/glotzerlab/signac) can be used for reference.

### Citation

Please cite this GitHub repository.

 - This repository:  Add repository here

### Installation

These signac workflows "this project" can be built using conda:

```bash
cd signac_numpy_tutorial
```

```bash
conda env create -f environment.yml
```

```bash
conda activate signac_numpy_tutorial
```

### Run the Workflow Locally

All commands in this section are run from the `<local_path>/signac_numpy_tutorial/signac_numpy_tutorial/project` directory.

Initialize all the state points for the jobs (generate all the separate folders with the same variables).  
 - Note: This command generates the `workspace` folder, which includes a sub-folder for each state point (different variable combinations),  These sub-folders are numbered uniquely based of the state point values.  The user can add more state points via the `init.py` file at any time, running the below command to create the new state points files and sub-folders that are in the `init.py` file.

 ```bash
python init.py
```

Check the status of your project (i.e., what parts are completed and what parts are available to be run).

```bash
python project.py status
```

Run `all available jobs for the whole project` locally with the `run` command.  Note: Using the run command like this will run all parts of the projects until completion.  Note: This feature is not available when submitting to HPCs.

```bash
python project.py run
```

Run all available `part 1` sections of the project locally with the `run` command.

```bash
python project.py run -o part_1_initial_parameters_command
```

Run all available `part 2` sections of the project locally with the `run` command.

```bash
python project.py run -o part_2_write_numpy_input_command
```

Run all available `part 3` sections of the project locally with the `run` command.

```bash
python project.py run -o part_3_numpy_calcs_command
```

Run all available `part 4` sections of the project locally with the `run` command.

```bash
python project.py run -o part_4_analysis_replicate_averages_command
```

Additionally, you can run the following flags for the  `run` command, controlling the how the jobs are executed on the local machine (does not produce HPC job submission scripts):
 - `--parallel 2` : This only works this way when using `run`. This runs several jobs in parallel (2 in this case) at a time on the local machine, auto adjusting the time, CPU cores, etc., based on the total command selections.
 - See the `signac` [documenation](https://docs.signac.io/en/latest/) for more information, features, and the [Project Command Line Interface](https://docs.signac.io/projects/flow/en/latest/project-cli.html).


### Submit the Workflow Jobs to an HPC.  

All commands in this section are run from the `<local_path>/signac_numpy_tutorial/signac_numpy_tutorial/project` directory.

First, you need to be sure that the `templates/phoenix.sh` or the used HPC template file is correct for the given HPC.  Additionally, the `templates/phoenix.sh` file is correct for the given HPC in the `project.py` file, specifically it is setup for  the `DefaultSlurmEnvironment` (only for a Slurm enviroment), and the class for it is set properly (Example: `class Phoenix(DefaultSlurmEnvironment):`).  

Second, in general, the `signac labels` (Example: `@Project.label` in the `project.py` file) that check the status of each workflow part should not be written in a way that is computationally expensive, removing the need to run an interactive job on the HPC when using the `signac status` command.  Otherwise, you need to run an interactive job when using the `signac status` command on the HPC, as it will be computationally expensive. 

Initialize all the state points for the jobs (generate all the separate folders with the different state points).  
 - Note: This command generates the `workspace` folder, which includes a sub-folder for each state point (different variable combinations),  These sub-folders are numbered uniquely based of the state point values.  The user can add more state points via the `init.py` file at any time, running the below command to create the new state points files and sub-folders that are in the `init.py` file.

 ```bash
python init.py
```

Check the status of your project (i.e., what parts are completed and what parts are available to be run).

```bash
python project.py status
```

Submit `all the currently available jobs` to the HPC with the `submit` command.

```bash
python project.py submit
```

Submit all available `part 1` sections of the project to the HPC with the `submit` command.

```bash
python project.py submit -o part_1_initial_parameters_command
```

Submit all available `part 2` sections of the project to the HPC with the `submit` command.

```bash
python project.py submit -o part_2_write_numpy_input_command
```

Submit all available `part 3` sections of the project to the HPC with the `submit` command.

```bash
python project.py submit -o part_3_numpy_calcs_command
```

Submit all available `part 4` sections of the project to the HPC with the `submit` command.

```bash
python project.py submit -o part_4_analysis_replicate_averages_command
```

Additionally, you can run the following flags for the `submit` command, controlling the how the jobs are submitted to the HPC:
 - `--bundle 2` : Only available when using `submit`.  This bundles multiple jobs (2 in this case) into a single run or HPC submittion script, auto adjusting the time, CPU cores, etc., based on the total command selections.
  - `--pretend` : Only available when using `submit`.  This is used to output what the submission script will look like, without submitting it to the HPC. 
  - `--parallel` : This only works this way when using `submit`.  The `N` value in `--parallel N` is not readl; therefore, it only runs all the jobs in a HPC submittion script at the same time (in parallel), auto adjusting the time, CPU cores, etc., based on the total command selections. 
  - See the `signac` [documenation](https://docs.signac.io/en/latest/) for more information, features, and the [Project Command Line Interface](https://docs.signac.io/projects/flow/en/latest/project-cli.html).