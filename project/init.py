"""Initialize signac statepoints."""

import os
import signac

# *******************************************
# ENTER THE MAIN USER STATEPOINTS (START)
# *******************************************
# Initialize the signac project
signac.init_project()

# Tested NUM_EPOCHS list [From: NUM_EPOCHS = int(os.environ.get("PLM_NUM_EPOCHS", 3))]
num_epochs_int_list = [3, 6]

# Tested BATCH_SIZE list [From: int(os.environ.get("PLM_BATCH_SIZE", 256))]
batch_size_int_list = [256, 512]

# Tested HIDDEN_SIZE list [From: int(os.environ.get("PLM_HIDDEN_SIZE", 64))]
hidden_size_int_list = [64]

# Tested LEARNING_RATE list [From: float(os.environ.get("PLM_LEARNING_RATE", 2e-4))]
learning_rate_float_list = [2e-4]

# Tested DROPOUT_PROB list [From: float(os.environ.get("PLM_DROPOUT_PROB", 0.1))]
dropout_prob_float_list = [0.1]

# Tested FGSM_EPSILON list [From: float(os.environ.get("PLM_FGSM_EPSILON", 0.05))]
fgsm_epsilon_float_list = [0.05]

# Tested SEED list [From: int(os.environ.get("PLM_SEED", 42))]
seed_int_list = [1, 2]


# *******************************************
# ENTER THE MAIN USER STATEPOINTS (END)
# *******************************************

# Setup the directories in the current directory
print("os.getcwd() = " +str(os.getcwd()))
pr_root = os.getcwd()
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
    pr.open_job(
        statepoint=sp,
    ).init()
