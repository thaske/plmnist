"""Basic example of a signac project"""
# project.py

import os
import numpy as np

import flow
from flow import FlowProject, aggregator
from flow.environment import DefaultSlurmEnvironment
import hpc_setup

# ******************************************************
# ******************************************************
# ******************************************************
# SIGNAC'S STARTING CODE SECTION (START)
# ******************************************************
# ******************************************************
# ******************************************************

class Project(FlowProject):
    """Subclass of FlowProject which provides the attributes and custom methods."""

    def __init__(self):
        super().__init__()


# ******************************************************
# ******************************************************
# ******************************************************
# SIGNAC'S STARTING CODE SECTION (END)
# ******************************************************
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# ******************************************************
# TYPICAL USER VARIBLES THAT CHANGE (START)
# ******************************************************
# ******************************************************
# ******************************************************

# file names extensions are added later 
# NOTE: DO NOT CHANGE NAMES AFTER STARTING PROJECT, ONLY AT THE BEGINNING
numpy_input_filename_str = "numpy_input_file"
numpy_output_filename_str = "numpy_output_file"
output_avg_std_of_replicates_txt_filename = "output_avg_std_of_replicates_txt_filename"

# Set the walltime, memory, and number of CPUs and GPUs needed
# for each individual job, based on the part/section.
part_1_walltime_hr = 0.25
part_1_memory_gb = 4
part_1_cpu_int = 1
part_1_gpu_int = 0

part_2_walltime_hr = 0.5
part_2_memory_gb = 4
part_2_cpu_int = 1
part_2_gpu_int = 0

part_3_walltime_hr = 0.75
part_3_memory_gb = 4
part_3_cpu_int = 1
part_3_gpu_int = 1

part_4_walltime_hr = 1
part_4_memory_gb = 4
part_4_cpu_int = 1
part_4_gpu_int = 0

# ******************************************************
# ******************************************************
# ******************************************************
# TYPICAL USER VARIBLES THAT CHANGE (END)
# ******************************************************
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# ******************************************************
# SIGNAC MAIN CODE SECTION (START)
# ******************************************************
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# CREATE THE INITIAL VARIABLES, WHICH WILL BE STORED IN 
# EACH JOB (START)
# ******************************************************
# ******************************************************

# SET THE PROJECTS DEFAULT DIRECTORY
project_directory = f"{os.getcwd()}"
print(f"project_directory = {project_directory}")


@Project.label
def part_1_initial_parameters_completed(job):
    """Check that the data is generated and in the json files."""
    data_written_bool = False
    if job.isfile(f"{'signac_job_document.json'}"):
        data_written_bool = True

    return data_written_bool


@Project.post(part_1_initial_parameters_completed)
@Project.operation(directives=
    {
        "np": part_1_cpu_int,
        "ngpu": part_1_gpu_int,
        "memory": part_1_memory_gb,
        "walltime": part_1_walltime_hr,
    }, with_job=True
)
def part_1_initial_parameters_command(job):
    """Set the system's job parameters in the json file."""
    
    # Note: the sp=setpoint variables (from init.py file), doc=user documented variables

    # Creating a new json file with user built variables (doc)
    job.doc.value_0_int = job.sp.value_0_int

    # Manipulating the set points (sp) variables from the init.py file and making new variables
    # Here we are simply adding 1, 2, and 3. 
    # NOTE: Values mported as strings, so they need converted to integers
    job.doc.value_1_int = int(int(job.sp.value_0_int) + 1)
    job.doc.value_2_int = int(int(job.sp.value_0_int) + 2)
    job.doc.value_3_int = int(int(job.sp.value_0_int) + 3)  

    # Print the replicate number on the .doc file also
    job.doc.replicate_number_int = job.sp.replicate_number_int

# ******************************************************
# ******************************************************
# CREATE THE INITIAL VARIABLES, WHICH WILL BE STORED IN 
# EACH JOB (END)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# FUNCTIONS ARE FOR GETTTING AND AGGREGATING DATA (START)
# ******************************************************
# ******************************************************

# Replaces runs can be looped together to average as needed
def statepoint_without_replicate(job):
    keys = sorted(tuple(j for j in job.sp.keys() if j not in {"replicate_number_int"}))
    return [(key, job.sp[key]) for key in keys]

# ******************************************************
# ******************************************************
# FUNCTIONS ARE FOR GETTTING AND AGGREGATING DATA (END)
# ******************************************************
# ******************************************************



# ******************************************************
# ******************************************************
# CREATE THE NUMPY FILE TO READ OR INPUT FILE (START)
# ******************************************************
# ******************************************************

# check if numpy input files are written
@Project.label
def part_2_write_numpy_input_written(job):
    """Check that the numpy input files are written ."""
    file_written_bool = False
    if (
        job.isfile(f"{numpy_input_filename_str}.txt")):
        file_written_bool = True

    return file_written_bool

@Project.pre(part_1_initial_parameters_completed)
@Project.post(part_2_write_numpy_input_written)
@Project.operation(directives=
    {
        "np": part_2_cpu_int,
        "ngpu": part_2_gpu_int,
        "memory": part_2_memory_gb,
        "walltime": part_2_walltime_hr,
    }, with_job=True
)
def part_2_write_numpy_input_command(job):
    """write the numpy input"""

    # get the integer values in the created json file with a random number generater
    numpy_input_value_0_int = int(int(job.doc.value_0_int) * 
                                  np.random.default_rng(seed=int(job.doc.replicate_number_int) + 1).integers(1, high=1000))
    numpy_input_value_1_int = int(int(job.doc.value_1_int) * 
                                  np.random.default_rng(seed=int(job.doc.replicate_number_int) + 2).integers(1, high=1000))
    numpy_input_value_2_int = int(int(job.doc.value_2_int) *
                                  np.random.default_rng(seed=int(job.doc.replicate_number_int) + 3).integers(1, high=1000))
    numpy_input_value_3_int = int(int(job.doc.value_3_int) * 
                                  np.random.default_rng(seed=int(job.doc.replicate_number_int) + 4).integers(1, high=1000))

    init_numpy_file = open(f"{numpy_input_filename_str}.txt", "w")
    init_numpy_file.write('{: <20} {: <20} {: <20} {: <20}'.format(
        numpy_input_value_0_int,
        numpy_input_value_1_int,
        numpy_input_value_2_int,
        numpy_input_value_3_int,
        )
    )
    init_numpy_file.close()


# ******************************************************
# ******************************************************
# CREATE THE NUMPY FILE TO READ OR INPUT FILE  (END)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# PERFORM THE NUMPY CALCULATIONS (START)
# ******************************************************
# ******************************************************

# check to see if the numpy calculations started
@Project.label 
def part_3a_numpy_calcs_started(job):
    """Check to see if the numpy calculations started."""
    output_started_bool = False
    if job.isfile(f"{numpy_output_filename_str}.txt"):
        output_started_bool = True

    return output_started_bool


# check to see if the numpy calculations completed correctly
@Project.label
def part_3b_numpy_calcs_completed_properly(job):
    """Check if the numpy calcs completed properly."""
    job_run_properly_bool = False
    output_log_file = f"{numpy_output_filename_str}.txt"
    if job.isfile(output_log_file):
        with open(job.fn(output_log_file), "r") as fp:
            output_line = fp.readlines()
            for i, line in enumerate(output_line):
                split_move_line = line.split()
                if "Numpy" in line and len(split_move_line) == 3:
                    if (
                        split_move_line[0] == "Numpy"
                        and split_move_line[1] == "Calculations"
                        and split_move_line[2] == "Completed"
                    ):
                        job_run_properly_bool = True
    else:
        job_run_properly_bool = False

    return job_run_properly_bool


@Project.pre(part_2_write_numpy_input_written)
@Project.post(part_3a_numpy_calcs_started)
@Project.post(part_3b_numpy_calcs_completed_properly)
@Project.operation(directives=
    {
        "np": part_3_cpu_int,
        "ngpu": part_3_gpu_int,
        "memory": part_3_memory_gb,
        "walltime": part_3_walltime_hr,
    }, with_job=True, cmd=True
)
def part_3_numpy_calcs_command(job):
    """Run the numpy calculations and any other bash command."""

    # If any previous replicate averages and std_devs exist delete them, 
    # because they will need recalculated as more state points were added.
    if os.path.isfile(f'../../analysis/{output_avg_std_of_replicates_txt_filename}.txt'):
        os.remove(f'../../analysis/{output_avg_std_of_replicates_txt_filename}.txt')

    # Read the numpy input file and conduct numpy calculation.
    # Put the 4 input numbers in an array,calcuate the dot product 
    # (4 input numbers in an array dot [1, 2, 3, 4]).
    # All output values are integers.
    input_file = f"{numpy_input_filename_str}.txt"
    with open(job.fn(input_file), "r") as fp:
        input_line = fp.readlines()
        split_input_line = input_line  
        for i, line in enumerate(input_line):
            split_input_line = line.split() 
            if len(split_input_line) == 4:  
                try:
                    print(f'****************')
                    print(f'****************')
                    print(f'****************')
                    print(f'****************')
                    input_array = np.array(
                        [
                        int(split_input_line[0]), 
                        int(split_input_line[1]), 
                        int(split_input_line[2]), 
                        int(split_input_line[3])
                        ],
                        dtype=int
                    ) 
                    print(f'input_array = {input_array}')
                    multiply_array = np.array([1, 2, 3, 4])
                    print(f'multiply_array = {multiply_array}')
                    dot_product = int(np.dot(input_array, multiply_array))
                    print(f'dot_product = {dot_product}')

                except:
                    raise ValueError("ERROR: The numpy input file is not 4 integer values.")


    # Write the output
    ouput_filename = open(f"{numpy_output_filename_str}.txt", "w")
    ouput_filename.write('{: <20}\n'.format(dot_product))
    ouput_filename.write('{: <20} {: <20} {: <20}'.format("Numpy", "Calculations", "Completed"))
    ouput_filename.close()


    # example of running a bash command
    print(f"Running job id {job}")
    run_command = "echo {}".format(
        'Running the echo command or any other bash command here',
    )

    print(f'example bash run command = {run_command}')

    return run_command


# ******************************************************
# ******************************************************
# PERFORM THE NUMPY CALCULATIONS (END)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# DATA ANALSYIS: GET THE REPLICATE DATA AVG AND STD. DEV (START)
# ******************************************************
# ******************************************************

# Check if the average and std. dev. of all the replicates is completed
@Project.label
def part_4_analysis_replica_averages_completed(*jobs):
    """Check that the replicate numpy averages files are written ."""
    file_written_bool_list = []
    all_file_written_bool_pass = False
    for job in jobs:
        file_written_bool = False

        if (
            job.isfile(
                f"../../analysis/{output_avg_std_of_replicates_txt_filename}.txt"
            )
        ):
            file_written_bool = True

        file_written_bool_list.append(file_written_bool)

    if False not in file_written_bool_list:
        all_file_written_bool_pass = True

    return all_file_written_bool_pass


@Project.pre(lambda *jobs: all(part_3b_numpy_calcs_completed_properly(j)
                               for j in jobs[0]._project))
@Project.post(part_4_analysis_replica_averages_completed)
@Project.operation(directives=
     {
         "np": part_4_cpu_int,
         "ngpu": part_4_gpu_int,
         "memory": part_4_memory_gb,
         "walltime": part_4_walltime_hr,
     }, aggregator=aggregator.groupby(key=statepoint_without_replicate, sort_by="value_0_int", sort_ascending=False)
)
def part_4_analysis_replicate_averages_command(*jobs):
    # Get the individial averages of the values from each state point,
    # and print the values in each separate folder.    


    # List the output column headers
    output_column_value_0_int_input_title = 'value_0_int' 
    output_column_numpy_avg_title = 'numpy_avg'
    output_column_numpy_std_dev_title = 'numpy_std_dev'  

    # create the lists for avg and std dev calcs
    value_0_int_repilcate_list = []
    numpy_replicate_list = []
    

    # write the output file before the for loop, so it gets all the 
    # values in the loops
    output_txt_file_header = \
        f"{output_column_value_0_int_input_title: <20} " \
        f"{output_column_numpy_avg_title: <20} " \
        f"{output_column_numpy_std_dev_title: <20} " \
        f" \n"

    write_file_name_and_path = f'analysis/{output_avg_std_of_replicates_txt_filename}.txt' 
    if os.path.isfile(write_file_name_and_path):
        replicate_calc_txt_file = open(write_file_name_and_path, "a")
    else:
        replicate_calc_txt_file = open(write_file_name_and_path, "w")
        replicate_calc_txt_file.write(output_txt_file_header)

    # Loop over all the jobs that have the same "value_0_int" (in sort_by="value_0_int"). 
    for job in jobs:
        # get the individual values
        output_file = f"{numpy_output_filename_str}.txt"
        with open(job.fn(output_file), "r") as fp:
            output_line = fp.readlines()
            split_output_line = output_line  
            for i, line in enumerate(output_line):
                split_line = line.split() 
                if len(split_line) == 1:
                   value_0_int_repilcate_list.append(int(job.doc.value_0_int)) 
                   numpy_replicate_list.append(int(split_line[0])) 
                

                elif not (
                    len(split_line) == 3 
                      and split_line[0] == 'Numpy' 
                      and split_line[1] == 'Calculations'
                      and split_line[2] == 'Completed'
                    ):
                    raise ValueError("ERROR: The format of the numpy output files are wrong.")

    # Check that the value_0_int are all the same and the aggregate function worked, 
    # grouping all the replicates of value_0_int
    for j in range(0, len(value_0_int_repilcate_list)):
        if value_0_int_repilcate_list[0] != value_0_int_repilcate_list[j]:
            raise ValueError(
                "ERROR: The value_0_int values are not grouping properly in the aggregate function."
                )
        value_0_int_aggregate = value_0_int_repilcate_list[0] 

    # Calculate the means and standard devs
    print(f'********************')
    print(f'value_0_int_aggregate = {value_0_int_aggregate}')
    print(f'********************')
    print(f'********************')
    print(f'numpy_replicate_list = {numpy_replicate_list}')

    numpy_avg = np.mean(numpy_replicate_list)
    numpy_avg_std = np.std(numpy_replicate_list, ddof=1)

   
    replicate_calc_txt_file.write(
        f"{value_0_int_aggregate: <20} "
        f"{numpy_avg: <20} "
        f"{numpy_avg_std: <20} "
        f" \n"
    )

    replicate_calc_txt_file.close()


# ******************************************************
# ******************************************************
# # DATA ANALSYIS: GET THE REPLICATE DATA AVG AND STD. DEV (END)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# ******************************************************
# SIGNAC MAIN CODE SECTION (END)
# ******************************************************
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# ******************************************************
# SIGNACS'S ENDING CODE SECTION (START)
# ******************************************************
# ******************************************************
# ******************************************************
if __name__ == "__main__":
    pr = Project()
    pr.main()
# ******************************************************
# ******************************************************
# ******************************************************
# SIGNACS'S ENDING CODE SECTION (END)
# ******************************************************
# ******************************************************
# ******************************************************