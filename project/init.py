"""Initialize signac statepoints."""

import signac
from pathlib import Path

# *******************************************
# ENTER THE MAIN USER STATEPOINTS (START)
# *******************************************
# Initialize the signac project
signac.init_project()

num_epochs_int_list = [5, 10]
batch_size_int_list = [128]
hidden_size_int_list = [64]
learning_rate_float_list = [2e-4]
dropout_prob_float_list = [0.0, 0.1, 0.5]
fgsm_epsilon_float_list = [0.05]
seed_int_list = [1, 2, 3]


# *******************************************
# ENTER THE MAIN USER STATEPOINTS (END)
# *******************************************

# Setup the directories in the current directory
pr_root = Path().absolute()
pr = signac.get_project(pr_root)


# Set all the statepoints, which will be used to create separate folders
# for each combination of state points.
all_statepoints = list()

for num_epochs_int_i in num_epochs_int_list:
    for batch_size_int_i in batch_size_int_list:
        for hidden_size_int_i in hidden_size_int_list:
            for learning_rate_float_i in learning_rate_float_list:
                for dropout_prob_float_i in dropout_prob_float_list:
                    for fgsm_epsilon_float_i in fgsm_epsilon_float_list:
                        for seed_int_i in seed_int_list:
                            statepoint = {
                                "num_epochs_int": num_epochs_int_i,
                                "batch_size_int": batch_size_int_i,
                                "hidden_size_int": hidden_size_int_i,
                                "learning_rate_float": learning_rate_float_i,
                                "dropout_prob_float": dropout_prob_float_i,
                                "fgsm_epsilon_float": fgsm_epsilon_float_i,
                                "seed_int": seed_int_i,
                            }

                            all_statepoints.append(statepoint)

# Initiate all statepoint createing the jobs/folders.
for sp in all_statepoints:
    pr.open_job(statepoint=sp).init()
