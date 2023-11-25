"""Signac implementation for the plmnist project."""

import json, datetime
import numpy as np
from pathlib import Path

import hpc_setup
from flow import FlowProject, aggregator
from signac.job import Job
from project.src.aggregate_analysis import write_output, statepoint_without_seed


# setup paths
signac_directory = Path.cwd()

if signac_directory.name != "signac":
    raise ValueError("Please run this script from inside the signac directory.")

root_directory = signac_directory.parent

data_directory = root_directory / "data"
output_file = signac_directory / "analysis" / "output.txt"

# ┌──────────────────────────────────┐
# │ Part 1 - write the job document. │
# └──────────────────────────────────┘


# post-condition: the job document has been written
@FlowProject.label
def part_1_initial_parameters_completed(job: Job):
    """Check that the job document has been written."""
    return job.isfile("signac_job_document.json")


# operation: write the job document
@FlowProject.post(part_1_initial_parameters_completed)
@FlowProject.operation(
    directives=dict(walltime=0.1, memory=4, np=1, ngpu=0),
    with_job=True,
)
def part_1_initial_parameters_command(job: Job):
    """Set the system's job parameters in the json file."""
    output_file.unlink(missing_ok=True)

    # here we write "signac_job_document.json" file.
    # signac takes care of the writing - we just add attributes to job.document
    # note that this file isn't used by the project - it's just here for demonstration.

    # note that we can access the statepoint variables (from init.py) via job.statepoint
    job.document.start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    job.document.seed = job.statepoint.seed_int


# ┌──────────────────────────────────┐
# │ Part 2 - download the MNIST data │
# └──────────────────────────────────┘


# post-condition: the data has been downloaded
@FlowProject.label
def part_2_download_the_dataset_completed(*job):
    """Check that data has been downloaded."""
    return data_directory.exists() and (data_directory / "MNIST").exists()


# operation: download the data
@FlowProject.post(part_2_download_the_dataset_completed)
@FlowProject.operation(
    directives=dict(walltime=0.2, memory=4, np=1, ngpu=0),
    cmd=True,
)
def part_2_download_the_dataset_command(*jobs):
    """Download the data."""
    data_directory.mkdir(parents=True, exist_ok=True)

    # Download the data set via a bash command
    return f"python -m plmnist.download --data_dir {data_directory}"


# ┌──────────────────────────────────────┐
# │ Part 3 - train and test a neural net │
# └──────────────────────────────────────┘


# post-condition: the results file exists
@FlowProject.label
def part_3_train_test_write_completed(job: Job):
    """Check if the results file exists."""
    return job.isfile("results.json")


# operation: run the train + test command
@FlowProject.pre(part_2_download_the_dataset_completed)
@FlowProject.post(part_3_train_test_write_completed)
@FlowProject.operation(
    directives=dict(walltime=1.0, memory=4, np=1, ngpu=1),
    with_job=True,
    cmd=True,
)
def part_3_train_test_write_command(job: Job):
    """Run the train + test command."""
    output_file.unlink(missing_ok=True)

    print(f"Running training/testing for {job}")
    return (
        f"python -m plmnist "
        f"--num_epochs {int(job.statepoint.num_epochs_int)} "
        f"--log_path {job.path} "
        f"--result_path {job.path} "
        f"--data_dir {data_directory} "
        f"--batch_size {int(job.statepoint.batch_size_int)} "
        f"--hidden_size {int(job.statepoint.hidden_size_int)} "
        f"--learning_rate {float(job.statepoint.learning_rate_float)} "
        f"--dropout_prob {float(job.statepoint.dropout_prob_float)} "
        f"--seed {int(job.statepoint.seed_int)} "
        f"--fgsm_epsilon {float(job.statepoint.fgsm_epsilon_float)} "
        f"--no_dhash "
        f"--no_fgsm "
    )


# ┌──────────────────────────────┐
# │ Part 4 - run the FGSM attack │
# └──────────────────────────────┘


# post-condition: the results contains the fgsm keys
@FlowProject.label
def part_4_fgsm_attack_completed(job: Job):
    """Check if the training, testing, and writing are completed properly."""
    if not job.isfile("results.json"):
        return False
    else:
        with open(job.fn("results.json"), "r") as json_log_file:
            try:
                loaded_json_file = json.load(json_log_file)
            except json.decoder.JSONDecodeError:
                return False

            for key in ["fgsm", "test_loss", "test_acc"]:
                if key not in loaded_json_file:
                    return False
                elif key == "fgsm":
                    if "accuracy" not in loaded_json_file["fgsm"]:
                        return False

            return True


# operation: run the fgsm attack command
@FlowProject.pre(part_3_train_test_write_completed)
@FlowProject.post(part_4_fgsm_attack_completed)
@FlowProject.operation(
    directives=dict(walltime=0.5, memory=4, np=1, ngpu=0),
    with_job=True,
    cmd=True,
)
def part_4_fgsm_attack_command(job: Job):
    """Run FGSM attack command."""
    output_file.unlink(missing_ok=True)

    print(f"Running fgsm for {job}")
    return (
        f"python -m plmnist.fgsm "
        f"--seed {int(job.statepoint.seed_int)} "
        f"--result_path {job.path} "
        f"--fgsm_epsilon {float(job.statepoint.fgsm_epsilon_float)} "
    )


# ┌─────────────────────────────────────┐
# │ Part 5 - compute avg/std over seeds │
# └─────────────────────────────────────┘


# post-condition: the replicate (seed) average file has been written
@FlowProject.label
def part_5_analysis_seed_averages_completed(*jobs):
    """Check that the replicate (seed) average file has been written."""
    return output_file.exists()


# operation: write the output file with the seed averages
@FlowProject.pre(lambda *jobs: all(part_4_fgsm_attack_completed(j) for j in jobs))
@FlowProject.post(part_5_analysis_seed_averages_completed)
@FlowProject.operation(
    directives=dict(walltime=0.6, memory=4, np=1, ngpu=0),
    aggregator=aggregator.groupby(
        key=statepoint_without_seed, sort_by="seed_int", sort_ascending=False
    ),
)
def part_5_analysis_seed_averages_command(*aggregated_jobs: Job):
    """Write the output file with the seed averages."""
    write_output(aggregated_jobs, output_file)


if __name__ == "__main__":
    try:
        pr = FlowProject()
    except LookupError as e:
        raise LookupError("Could not find project. Did you run init.py?") from e

    pr.main()
