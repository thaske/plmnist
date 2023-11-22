"""Setup the HPC and custom template location"""
# hpc_setup.py

from flow.environment import DefaultSlurmEnvironment

class Phoenix(DefaultSlurmEnvironment):  
    """Subclass of DefaultSlurmEnvironment for GT Phoenix HPC."""
    
    # Find the hostname by loggin in the HPC and using the 'hostname' command.
    # In this case 'hostname' produced 'login-phoenix-slurm-2.pace.gatech.edu'.
    hostname_pattern = r"login-phoenix-slurm-.\.pace\.gatech\.edu"
    template = "phoenix.sh"