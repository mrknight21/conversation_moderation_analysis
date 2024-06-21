#!/bin/bash
#SBATCH --partition=bigmem
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=mingbin.longformer
#SBATCH --cpus-per-task=16
#SBATCH -A punim0478
#SBATCH --mem=240G
#SBATCH --mail-user=mingbinc@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

# Add the export command here
export CUDA_LAUNCH_BLOCKING=1

source /data/projects/punim0478/bryan/anaconda3/bin/activate gpu_env
python3 main.py "$@"

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

