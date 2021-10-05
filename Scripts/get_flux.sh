#!/bin/bash

#SBATCH --job-name=flux

# Request CPU resource for a serial job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
# SBATCH --partition=short,comp

# Memory usage (MB)
#SBATCH --mem=30G

# Set your minimum acceptable walltime, format: day-hours:minutes:seconds
#SBATCH --time=10:00:00

#SBATCH --mail-user=csmi0005@student.monash.edu
#SBATCH --mail-type=FAIL

# >>> conda initialize >>> SPECIFIC TO MONARCH
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/anaconda/2020.07-python3.8-gcc8/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/anaconda/2020.07-python3.8-gcc8/etc/profile.d/conda.sh" ]; then
        . "/usr/local/anaconda/2020.07-python3.8-gcc8/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/anaconda/2020.07-python3.8-gcc8/bin:$PATH"
    fi
fi
unset __conda_setup
conda config --add pkgs_dirs /home/csmi0005/Mona0028/csmi0005/conda/pkgs
# <<< conda initialize <<<#

# activate conda environment
conda activate ./Data_env

model="UV_GAN_1"
iters=("0050000" "0100000" "0150000" "0200000" "0250000" "0300000" "0350000" "0400000" "0450000" "0500000")

    echo 'Getting flux for hmi.np_path_normal'
python Data_processing/get_unsigned_flux.py \
    --data 'hmi.np_path_normal'

# flux according to UV GAN
for iter in ${iters[@]}
do
    echo 'Getting flux for'
    echo 'hmi.np_path_normal' "aia.${model}_iter_${iter}_path" "euvi.${model}_iter_${iter}_path"
    python Data_processing/get_unsigned_flux.py \
        --data 'hmi.np_path_normal' "aia.${model}_iter_${iter}_path" "euvi.${model}_iter_${iter}_path"
done