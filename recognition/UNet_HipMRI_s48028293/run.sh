#!/bin/bash
# Usage:  sbatch run.sh                    (2D U-Net)
#         sbatch run.sh unet3d             (3D U-Net)
#         sbatch run.sh improved_unet3d    (3D Improved U-Net)
#SBATCH --job-name=unet_hipmri
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=unet_%j.out
#SBATCH --error=unet_%j.err

MODEL=${1:-unet2d}   # read the model name first
set --               # clear the arguments so conda's activate doesn't see "unet3d"
set -e               # stop if train.py fails, so predict.py doesn't run without a checkpoint

source $HOME/miniconda3/bin/activate
conda activate torch

OUT=../runs/${MODEL}_$SLURM_JOB_ID

python train.py --model $MODEL --out-dir $OUT
python predict.py --model $MODEL --out-dir $OUT --checkpoint $OUT/best_model.pt