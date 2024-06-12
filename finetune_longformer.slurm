#!/bin/bash
#SBATCH --partition=deeplearn
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:A100:2
#SBATCH --job-name=mingbin.longformer
#SBATCH --mem-per-cpu=48G
#SBATCH --qos gpgpudeeplearn
#SBATCH --mail-user=mingbinc@student.unimelb.edu.au
#SBATCH --constraint=dlg5
#SBATCH -A punim0521
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END



source /data/projects/punim0478/bryan/anaconda3/bin/activate gpu_env
python3 finetune_longformer.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

