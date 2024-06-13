#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --gres=gpu:A100:4
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --job-name=mingbin.longformer
#SBATCH --cpus-per-task=16
#SBATCH -A punim0521
#SBATCH --qos=gpgpudeeplearn
#SBATCH --constraint=dlg4
#SBATCH --mem=120G
#SBATCH --mail-user=mingbinc@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END

source /data/projects/punim0478/bryan/anaconda3/bin/activate gpu_env
python3 finetune_longformer.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

