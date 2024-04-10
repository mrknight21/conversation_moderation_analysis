#!/bin/bash
#SBATCH --partition=gpu-a100-short
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --mail-user=mingbinc@student.unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END


source /data/projects/punim0478/bryan/anaconda3/bin/activate gpu_env
python3 intervention_timing_classification.py --do_train --do_eval --max_seq_length 512 --per_device_train_batch_size 8 --learning_rate 2e-5 --num_train_epochs 20 --output_dir ./output/ --train_file ./data/isd_train.csv --validation_file ./data/isd_sampled_test.csv


##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s

