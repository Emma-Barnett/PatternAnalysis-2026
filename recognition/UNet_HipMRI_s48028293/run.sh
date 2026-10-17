#!/bin/bash
#SBATCH --job-name=unet_2d
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=unet_2d_%j.out
#SBATCH --error=unet_2d_%j.err
# Activate your conda environment
source $HOME/miniconda3/bin/activate
conda activate torch

OUT=../runs/unet_2d_$SLURM_JOB_ID

python train.py --out-dir $OUT
python predict.py --out-dir $OUT --checkpoint $OUT/best_model.pt