"""Basic example of a signac project"""
# project.py

import os
from flow import FlowProject, aggregator
import hpc_setup
from plmnist import train, test, write
from model import LitMNIST
from fgsm import fgsm_from_path, plot_fgsm
import json
import numpy as np

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


# Set the walltime, memory, and number of CPUs and GPUs needed
# for each individual job, based on the part/section.
part_1_walltime_hr = 0.1
part_1_memory_gb = 4
part_1_cpu_int = 1
part_1_gpu_int = 0

part_2_walltime_hr = 0.2
part_2_memory_gb = 4
part_2_cpu_int = 1
part_2_gpu_int = 0

part_3_walltime_hr = 1
part_3_memory_gb = 4
part_3_cpu_int = 1
part_3_gpu_int = 1

part_4_walltime_hr = 0.5
part_4_memory_gb = 4
part_4_cpu_int = 1
part_4_gpu_int = 0

part_5_walltime_hr = 0.6
part_5_memory_gb = 4
part_5_cpu_int = 1
part_5_gpu_int = 0

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
# SET THE PARAMETERS FOR SIGNAC 
# AND THE FILE/FOLDER NAMES (START)
# ******************************************************
# ******************************************************

# set data locations and names
downloaded_data_directory_name_relative_to_each_job = '../../data'
downloaded_data_directory_name_relative_to_project_py_file = 'data'
plmnist_directory_directory_name_relative_to_each_job = '../../../install_custom_package/plmnist'
plmnist_directory_directory_name_relative_to_project_py_file = '../install_custom_package/plmnist'
output_avg_std_of_seed_txt_filename = "output_avg_std_of_seed_txt_filename"

# SET THE PROJECTS DEFAULT DIRECTORY
project_directory = f"{os.getcwd()}"
print(f"project_directory = {project_directory}")

# ******************************************************
# ******************************************************
# SET THE PARAMETERS FOR SIGNAC 
# AND THE FILE/FOLDER NAMES (END)
# ******************************************************
# ******************************************************

# ******************************************************
# ******************************************************
# CREATE THE INITIAL VARIABLES, WHICH WILL BE STORED IN 
# EACH JOB (START)
# ******************************************************
# ******************************************************

# Part 1: Check the statepoints (individual jobs/runs) are generated
@Project.label
def part_1_initial_parameters_completed(job):
    """Check that the data is generated and in the json files."""
    data_written_bool = False
    if job.isfile(f"{'signac_job_document.json'}"):
        data_written_bool = True

    return data_written_bool

# Run Part 1
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
    # These do not need to be created fo the existing variables in 
    # 'job.sp', and the 'job.doc' be used to store any calculated values 
    # in the json file for later use, which can be done in this 
    # part or in other parts.
    job.doc.num_epochs_int = job.sp.num_epochs_int
    job.doc.batch_size_int = job.sp.batch_size_int
    job.doc.hidden_size_int = job.sp.hidden_size_int
    job.doc.learning_rate_float = job.sp.learning_rate_float
    job.doc.dropout_prob_float = job.sp.dropout_prob_float
    job.doc.fgsm_epsilon_float = job.sp.fgsm_epsilon_float
    job.doc.seed_int = job.sp.seed_int
    
    # Storing a calculated value for alter use
    job.doc.seed_times_batch_size_int = int(int(job.sp.seed_int) * int(job.sp.batch_size_int))

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

# Replaces runs can be looped together to average as needed, 
# with all the same variables, except the 'seed_int'.
def statepoint_without_seed(job):
    keys = sorted(tuple(j for j in job.sp.keys() if j not in {"seed_int"}))
    return [(key, job.sp[key]) for key in keys]

# ******************************************************
# ******************************************************
# FUNCTIONS ARE FOR GETTTING AND AGGREGATING DATA (END)
# ******************************************************
# ******************************************************



# ******************************************************
# ******************************************************
# DOWNLOAD THE DATASET (START)
# ******************************************************
# ******************************************************

# Part 2: Check the data set is downloaded
@Project.label
def part_2_download_the_dataset_completed(*job): 
    """Check that data is downloaded."""
    files_downloaded_bool = False

    # Look back 2 directories from the job folder, as for at least 1 file that was downloaded.
    # Note: yUou can check all files if you want by adding checks for all of them.
    check_one_data_file_relative_to_each_job = \
        f"{downloaded_data_directory_name_relative_to_project_py_file}/MNIST/raw/t10k-images-idx3-ubyte"
    if (
        os.path.isdir(downloaded_data_directory_name_relative_to_project_py_file) is True
        ):
        files_downloaded_bool = True

    return files_downloaded_bool

# Run Part 2 as only 1 job
@Project.pre(lambda *jobs: all(part_1_initial_parameters_completed(j) for j in jobs[0]._project))
@Project.post(part_2_download_the_dataset_completed)
@Project.operation(directives=
     {
         "np": part_2_cpu_int,
         "ngpu": part_2_gpu_int,
         "memory": part_2_memory_gb,
         "walltime": part_2_walltime_hr,
     }, cmd=True, aggregator=aggregator(select=lambda x: 1 > 0) # make only 1 job
)
def part_2_download_the_dataset_command(*jobs):  
    """write the plmnist input"""
    # make the data directory
    if os.path.isdir(f'{downloaded_data_directory_name_relative_to_project_py_file}') is False:
        os.mkdir(f'{downloaded_data_directory_name_relative_to_project_py_file}')

    # Download the data set via a bash command
    return f"python {plmnist_directory_directory_name_relative_to_project_py_file}/download.py " \
           f"--data_dir {downloaded_data_directory_name_relative_to_project_py_file}"

# ******************************************************
# ******************************************************
# DOWNLOAD THE DATASET (END)
# ******************************************************
# ******************************************************


# ******************************************************
# ******************************************************
# TRAIN, TEST, AND WRITE OUTPUT FILES (START)
# ******************************************************
# ******************************************************

# Part 3: check to see if the test, train, and write completed correctly
@Project.label
def part_3_train_test_write_completed(job):
    """Check if the training, testing, and writing are completed properly."""
    job_run_properly_bool = False
    output_log_file = f"results.json"
    if job.isfile(output_log_file):
        job_run_properly_bool = True

    return job_run_properly_bool

# Run Part 3
@Project.pre(part_2_download_the_dataset_completed)
@Project.post(part_3_train_test_write_completed)
@Project.operation(directives=
    {
        "np": part_3_cpu_int,
        "ngpu": part_3_gpu_int,
        "memory": part_3_memory_gb,
        "walltime": part_3_walltime_hr,
    }, with_job=True, cmd=True
)
def part_3_train_test_write_command(job):
    """Run the training, testing, and writing command."""

    # If any previous seed averages and std_devs exist delete them, 
    # because they will need recalculated as more state points were added.
    if os.path.isfile(f'../../analysis/{output_avg_std_of_seed_txt_filename}.txt'):
        os.remove(f'../../analysis/{output_avg_std_of_seed_txt_filename}.txt')

    # Run the the training, testing, and writing via a bash command, 
    # without fgsm (i.e., the '--no_fgsm' flag)
    print(f"Running job id {job}")
    return (
        f"python {plmnist_directory_directory_name_relative_to_each_job}/run_training_and_testing.py " 
          f"--num_epochs {int(job.sp.num_epochs_int)} "
          f"--log_path {'./'} "
          f"--result_path {'./'} "
          f"--data_dir {downloaded_data_directory_name_relative_to_each_job} "
          f"--batch_size {int(job.sp.batch_size_int)} "
          f"--hidden_size {int(job.sp.hidden_size_int)} "
          f"--learning_rate {float(job.sp.learning_rate_float)} "
          f"--dropout_prob {float(job.sp.dropout_prob_float)} "
          f"--seed {int(job.sp.seed_int)} "
          f"--fgsm_epsilon {float(job.sp.fgsm_epsilon_float)} "
          f"--no_dhash "
          f"--no_fgsm "
        )

# ******************************************************
# ******************************************************
# TRAIN, TEST, AND WRITE OUTPUT FILES  (END)
# ******************************************************
# ******************************************************



# ******************************************************
# ******************************************************
# TRAIN, TEST, AND WRITE OUTPUT FILES (START)
# ******************************************************
# ******************************************************

# Part 4: check to see if the test, train, and write completed correctly
@Project.label
def part_4_fgsm_attack_completed(job):
    """Check if the training, testing, and writing are completed properly."""
    job_run_properly_bool = False
    output_log_file = f"results.json"
    if job.isfile(output_log_file):
        with open(job.fn(output_log_file), "r") as json_log_file:
            loaded_json_file = json.load(json_log_file)

            try:
                # Check by verifying 'fgsm' is printed in the results file
                loaded_json_file['fgsm']
                loaded_json_file['fgsm']['accuracy']
                loaded_json_file["val_loss"]
                loaded_json_file["val_acc"]
                loaded_json_file["test_loss"]
                loaded_json_file["test_acc"]
                job_run_properly_bool = True
            
            except:
                job_run_properly_bool = False

    return job_run_properly_bool

# Run Part 4
@Project.pre(part_3_train_test_write_completed)
@Project.post(part_4_fgsm_attack_completed)
@Project.operation(directives=
    {
        "np": part_3_cpu_int,
        "ngpu": part_3_gpu_int,
        "memory": part_3_memory_gb,
        "walltime": part_3_walltime_hr,
    }, with_job=True, cmd=True
)
def part_4_fgsm_attack_command(job):
    """Run FGSM attack command."""

    # If any previous seed averages and std_devs exist delete them, 
    # because they will need recalculated as more state points were added.
    if os.path.isfile(f'../../analysis/{output_avg_std_of_seed_txt_filename}.txt'):
        os.remove(f'../../analysis/{output_avg_std_of_seed_txt_filename}.txt')

    # Run the the training, testing, and writing via a bash command, 
    # without fgsm (i.e., the '--no_fgsm' flag)
    print(f"Running job id {job}")
    return (
        f"python {plmnist_directory_directory_name_relative_to_each_job}/fgsm.py " 
        f"--seed {int(job.sp.seed_int)} "
        f"--result_path {'./'} "
        f"--fgsm_epsilon {float(job.sp.fgsm_epsilon_float)} "
        )

# ******************************************************
# ******************************************************
# TRAIN, TEST, AND WRITE OUTPUT FILES  (END)
# ******************************************************
# ******************************************************






# ******************************************************
# ******************************************************
# # DATA ANALSYIS: GET THE REPLICATE DATA 
# (Different seed numbers) AVG AND STD. DEV. (START)
# ******************************************************
# ******************************************************

# Part 5: Check if the average and std. dev. of all the replicate (seed) is completed
@Project.label
def part_5_analysis_seed_averages_completed(*jobs):
    """Check that the replicate (seed) averages files are written ."""
    file_written_bool_list = []
    all_file_written_bool_pass = False
    for job in jobs:
        file_written_bool = False

        if (
            job.isfile(
                f"../../analysis/{output_avg_std_of_seed_txt_filename}.txt"
            )
        ):
            file_written_bool = True

        file_written_bool_list.append(file_written_bool)

    if False not in file_written_bool_list:
        all_file_written_bool_pass = True

    return all_file_written_bool_pass

# Run Part 5
@Project.pre(lambda *jobs: all(part_4_fgsm_attack_completed(j)
                               for j in jobs[0]._project))
@Project.post(part_5_analysis_seed_averages_completed)
@Project.operation(directives=
     {
         "np": part_5_cpu_int,
         "ngpu": part_5_gpu_int,
         "memory": part_5_memory_gb,
         "walltime": part_5_walltime_hr,
     }, aggregator=aggregator.groupby(key=statepoint_without_seed, sort_by="seed_int", sort_ascending=False)
)
def part_5_analysis_seed_averages_command(*jobs):
    # Get the individial averages of the values from each state point,
    # and print the values in each separate folder.    


    # List the output column headers
    output_column_num_epochs_int_title = 'num_epochs_int' 
    output_column_batch_size_int_title = 'batch_size_int' 
    output_column_hidden_size_int_title = 'hidden_size_int' 
    output_column_learning_rate_float_title = 'learning_rate_float' 
    output_column_dropout_prob_float_title = 'dropout_prob_float' 
    output_column_fgsm_epsilon_float_title = 'fgsm_epsilon_float' 

    output_column_test_acc_avg_title = 'test_acc_avg' 
    output_column_test_loss_avg_title = 'test_loss_avg' 
    output_column_val_acc_avg_title = 'val_acc_avg' 
    output_column_val_loss_avg_title = 'val_loss_avg'
    output_column_fgsm_acc_avg_title = 'fgsm_acc_avg' 

    output_column_test_acc_std_dev_title = 'test_acc_std_dev' 
    output_column_test_loss_std_dev_title = 'test_loss_std_dev' 
    output_column_val_acc_std_dev_title = 'val_acc_std_dev' 
    output_column_val_loss_std_dev_title = 'val_loss_std_dev'
    output_column_fgsm_acc_std_dev_title = 'fgsm_acc_std_dev' 


    # create the lists for avg and std dev calcs
    num_epochs_int_list = []
    batch_size_int_list = []
    hidden_size_int_list = []
    learning_rate_float_list = []
    dropout_prob_float_list = []
    fgsm_epsilon_float_list = []

    test_acc_list = []
    test_loss_list = []
    val_acc_list = []
    val_loss_list = []
    fgsm_acc_list = []

    # write the output file before the for loop, so it gets all the 
    # values in the loops
    output_txt_file_header = \
        f"{output_column_num_epochs_int_title: <25} " \
        f"{output_column_batch_size_int_title: <25} " \
        f"{output_column_hidden_size_int_title: <25} " \
        f"{output_column_learning_rate_float_title: <25} " \
        f"{output_column_dropout_prob_float_title: <25} " \
        f"{output_column_fgsm_epsilon_float_title: <25} " \
        f"{output_column_test_acc_avg_title: <25} " \
        f"{output_column_test_acc_std_dev_title: <25} " \
        f"{output_column_test_loss_avg_title: <25} " \
        f"{output_column_test_loss_std_dev_title: <25} " \
        f"{output_column_val_acc_avg_title: <25} " \
        f"{output_column_val_acc_std_dev_title: <25} " \
        f"{output_column_val_loss_avg_title: <25} " \
        f"{output_column_val_loss_std_dev_title: <25} " \
        f"{output_column_fgsm_acc_avg_title: <25} " \
        f"{output_column_fgsm_acc_std_dev_title: <25} " \
        f" \n"

    write_file_name_and_path = f'analysis/{output_avg_std_of_seed_txt_filename}.txt' 
    if os.path.isfile(write_file_name_and_path):
        seed_calc_txt_file = open(write_file_name_and_path, "a")
    else:
        seed_calc_txt_file = open(write_file_name_and_path, "w")
        seed_calc_txt_file.write(output_txt_file_header)

    # Loop over all the jobs that have the same "seed_int" (in sort_by="seed_int"). 
    for job in jobs:
        # get the individual values
        output_log_file = f"results.json"
        if job.isfile(output_log_file):
            with open(job.fn(output_log_file), "r") as json_log_file:
                loaded_json_file = json.load(json_log_file)

                num_epochs_int_list.append(int(job.sp.num_epochs_int))
                batch_size_int_list.append(int(job.sp.batch_size_int))
                hidden_size_int_list.append(int(job.sp.hidden_size_int))
                learning_rate_float_list.append(float(job.sp.learning_rate_float))
                dropout_prob_float_list.append(float(job.sp.dropout_prob_float))
                fgsm_epsilon_float_list.append(float(job.sp.fgsm_epsilon_float))

                test_acc_list.append(float(loaded_json_file["test_acc"]))
                test_loss_list.append(float(loaded_json_file["test_loss"]))
                val_acc_list.append(float(loaded_json_file["val_acc"]))
                val_loss_list.append(float(loaded_json_file["val_loss"]))
                fgsm_acc_list.append(float(loaded_json_file['fgsm']['accuracy']))

    # Check that the 'seed_int' are all the same and the aggregate function worked, 
    # grouping all the seed of 'seed_int'
    for j in range(0, len(num_epochs_int_list)):
        if (num_epochs_int_list[0] != num_epochs_int_list[j] 
            or batch_size_int_list[0] != batch_size_int_list[j] 
            or hidden_size_int_list[0] != hidden_size_int_list[j] 
            or learning_rate_float_list[0] != learning_rate_float_list[j] 
            or dropout_prob_float_list[0] != dropout_prob_float_list[j] 
            or fgsm_epsilon_float_list[0] != fgsm_epsilon_float_list[j] 
            ):
            raise ValueError(
                "ERROR: The num_epochs_int_list, batch_size_int_list, hidden_size_int_list, "
                "learning_rate_float_list, dropout_prob_float_list, or fgsm_epsilon_float_list "
                "values are not grouping properly in the aggregate function."
                )
        # set the same values as aggreate values
        num_epochs_int_aggregate = num_epochs_int_list[0] 
        batch_size_int_aggregate = batch_size_int_list[0] 
        hidden_size_int_aggregate = hidden_size_int_list[0] 
        learning_rate_float_aggregate = learning_rate_float_list[0] 
        dropout_prob_float_aggregate = dropout_prob_float_list[0] 
        fgsm_epsilon_float_aggregate = fgsm_epsilon_float_list[0] 

    # Calculate the means and standard devs
    test_acc_avg = np.mean(test_acc_list)
    test_loss_avg = np.mean(test_loss_list)
    val_acc_avg = np.mean(val_acc_list)
    val_loss_avg = np.mean(val_loss_list)
    fgsm_acc_avg = np.mean(fgsm_acc_list)

    test_acc_std = np.std(test_acc_list, ddof=1)
    test_loss_std = np.std(test_loss_list, ddof=1)
    val_acc_std = np.std(val_acc_list, ddof=1)
    val_loss_std = np.std(val_loss_list, ddof=1)
    fgsm_acc_std = np.std(fgsm_acc_list, ddof=1)

    # write out all the values
    seed_calc_txt_file.write(
        f"{num_epochs_int_aggregate: <25} "
        f"{batch_size_int_aggregate: <25} "
        f"{hidden_size_int_aggregate: <25} "
        f"{learning_rate_float_aggregate: <25} "
        f"{dropout_prob_float_aggregate: <25} "
        f"{fgsm_epsilon_float_aggregate: <25} "
        f"{test_acc_avg: <25} "
        f"{test_acc_std: <25} "
        f"{test_loss_avg: <25} "
        f"{test_loss_std: <25} "
        f"{val_acc_avg: <25} "
        f"{val_acc_std: <25} "
        f"{val_loss_avg: <25} "
        f"{val_loss_std: <25} "
        f"{fgsm_acc_avg: <25} "
        f"{fgsm_acc_std: <25} "
        f" \n"
    )

    seed_calc_txt_file.close()

# ******************************************************
# ******************************************************
# DATA ANALSYIS: GET THE REPLICATE DATA 
# (Different seed numbers) AVG AND STD. DEV. (END)
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
