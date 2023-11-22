{% extends "slurm.sh" %}

{% block header %}
{% set gpus = operations|map(attribute='directives.ngpu')|sum %}
{% set memory = operations|map(attribute='directives.memory')|sum %}
{% set np = operations|map(attribute='directives.np')|sum %}
    {{- super () -}}

{% if gpus %}
#SBATCH -p gpu-a100
#SBATCH --gres gpu:{{ gpus }}

{%- else %}
#SBATCH -p cpu-small

{%- endif %}

#SBATCH -A phx-pace-staff
#SBATCH -N 1
#SBATCH -q inferno
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

echo  "Running on host" hostname
echo  "Time is" date

# load any modules here needed for both CPU and GPU versions
module anaconda3

# Add any modules here needed only for the GPU versions
{% if gpus %}
module load cuda/12.1.1-6oacj6
{%- endif %}

# activate the required conda environment
conda activate signac_numpy_example

{% endblock header %}

{% block body %}
    {{- super () -}}


{% endblock body %}