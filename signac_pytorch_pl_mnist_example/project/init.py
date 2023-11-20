"""Initialize signac statepoints."""

import os
import numpy as np
import signac

# *******************************************
# ENTER THE MAIN USER STATEPOINTS (START)
# *******************************************
# Initialize the signac project
signac.init_project()

# Enter the variable 'values' (integer_only) 
# values_int = [set_0_value_0, set_1_value_0]
values_int = [1, 2]

# Enter the number of replicates desired (replicate_number). 
# replicate_number = [0, 1, 2, 3, 4]
replicate_number = [0, 1]


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

for values_int_i in values_int:
    for replicate_i in replicate_number:
        statepoint = {
            "value_0_int": values_int_i,
            "replicate_number_int": replicate_i,
        }

        all_statepoints.append(statepoint)

# Initiate all statepoint createing the jobs/folders.
for sp in all_statepoints:
    pr.open_job(
        statepoint=sp,
    ).init()
