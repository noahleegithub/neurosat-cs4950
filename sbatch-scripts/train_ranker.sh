#!/bin/bash
#SBATCH --job-name=train_ranker
#SBATCH --time=08:00:00
#SBATCH --output=./slurm/slurm-%j.out
#SBATCH --error=./slurm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source activate FuzzyLogic

python train_ranker.py configurations/config_rank_pair_val.yaml

