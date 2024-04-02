{% extends "slurm.sh" %}

{% block header %}
{% set gpus = operations|map(attribute='directives.ngpu')|max %}
{% set mem_per_cpu = operations|map(attribute='directives.mem-per-cpu')|max  %}
{% set cpus_per_task = operations|map(attribute='directives.cpus-per-task')|max  %}

    {{- super () -}}

{% if gpus %}
#SBATCH -p gpu-a100
#SBATCH --gres gpu:{{ gpus }}

{%- else %}
#SBATCH -p cpu-small

{%- endif %}

#SBATCH -A phx-pace-staff
#SBATCH -N 1
#SBATCH --cpus-per-task={{ cpus_per_task }}
#SBATCH -q inferno
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --mem-per-cpu={{ mem_per_cpu }}G

echo  "Running on host" hostname
echo  "Time is" date

# load any modules here needed for both CPU and GPU versions
module load anaconda3

# Add any modules here needed only for the GPU versions
{% if gpus %}
module load cuda/12.1.1-6oacj6
{%- endif %}

# activate the required conda environment
conda activate plmnist

{% endblock header %}

{% block body %}
    {{- super () -}}


{% endblock body %}