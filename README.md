# Hyperparameter Tuning Tutorial: Learning MNIST with PyTorch Lightning and Ray Tune

This repository demonstrates how to use Ray Tune to perform hyperparameter tuning. The task in this case is to train a simple neural network on the MNIST dataset using PyTorch Lightning. This allows to easily scale from running locally on a laptop to running on a multi-node cluster.

## Installation

0. Install `conda` if not already installed.
<details><summary>Using Mambaforge is recommended</summary>

___
First download the appropriate installer script for your system:

| OS      | Architecture          | Download  |
| --------|-----------------------|-----------|
| Linux   | x86_64 (amd64)        | [Mambaforge-Linux-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-x86_64.sh) |
| Linux   | aarch64 (arm64)       | [Mambaforge-Linux-aarch64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-aarch64.sh) |
| Linux   | ppc64le (POWER8/9)    | [Mambaforge-Linux-ppc64le](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Linux-ppc64le.sh) |
| OS X    | x86_64                | [Mambaforge-MacOSX-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-x86_64.sh) |
| OS X    | arm64 (Apple Silicon) | [Mambaforge-MacOSX-arm64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-MacOSX-arm64.sh) |
| Windows | x86_64                | [Mambaforge-Windows-x86_64](https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-Windows-x86_64.exe) |

Then run the installer script and follow the instructions (replace `<path to script>` with the full path to the downloaded script):
```bash
bash <path to script>
```

In the commands below, replace `conda env create` with `mamba env create` for faster package installation.
___
</details>

1. Clone the repository and navigate to the directory:
```bash
git clone git@github.com:klamike/plmnist.git
cd plmnist
```

2. Install the required packages:
```bash
# for CPU-only:
conda env create -f environment.yml
# for CUDA GPU:
conda env create -f cuda_environment.yml
```

3. Activate the environment:
```bash
conda activate plmnist
```

4. Download the data:
```bash
python -m plmnist.download
```

## CPU Usage

Use these instructions if you're running on your laptop.

1. Start Ray
```bash
ray start --head
```

2. Run the hyperparameter tuning script:
```bash
python -m plmnist.tune
```

## SLURM Usage

Use these instructions if you're running on a single (potentially multi-GPU) node on SLURM.

0. Checkout the [`tune_slurm`](https://github.com/klamike/plmnist/tree/tune_slurm) branch:
```bash
git checkout tune_slurm
```

1. Start Ray
```bash
ray start --head
```

2. Run the hyperparameter tuning script:
```bash
python -m plmnist.tune
```


## Multi-Node Usage

Use these instructions if you're running on multiple nodes on SLURM. You'll need to reserve one CPU head node and a few GPU worker nodes.

0. Checkout the [`tune_slurm`](https://github.com/klamike/plmnist/tree/tune_slurm) branch:
```bash
git checkout tune_slurm
```

1. Generate a Redis password:
```bash
export redis_password = $(uuidgen)
echo $redis_password
```

2. Find the IP address of the head node:
```bash
hostname -I
```

3. Start Ray on the head node (replace `<redis password>` with the generated password):
```bash
ray start --head --redis-password=<redis password>
```

4. **On each worker node**, link the GPU worker to the head node (replace `<head node ip address>` and `<redis password>`):
```bash
ray start --address=<head node ip address> --redis-password=<redis password>
```

5. Run the hyperparameter tuning script on the head node:
```bash
python -m plmnist.tune
```
