#!/bin/bash
#SBATCH --job-name=train_sat.sh
#SBATCH --time=08:00:00
#SBATCH --output=./slurm/slurm-%j.out
#SBATCH --error=./slurm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source activate FuzzyLogic

python train_sat.py configurations/config_sat_yager5.yaml



