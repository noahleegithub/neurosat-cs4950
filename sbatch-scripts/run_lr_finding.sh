#!/bin/bash
#SBATCH --job-name=sgodel_sr_10_40
#SBATCH --time=08:00:00
#SBATCH --output=./slurm/slurm-%j.out
#SBATCH --error=./slurm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source activate FuzzyLogic
python -c "import torch; print(torch.cuda.is_available())"
python lr_finding.py config.yaml > lr_results_new_architecture.txt

