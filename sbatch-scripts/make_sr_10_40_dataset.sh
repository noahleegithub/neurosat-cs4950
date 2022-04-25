#!/bin/bash
#SBATCH --job-name=create_sr1040_data
#SBATCH --time=24:00:00
#SBATCH --output=./slurm/slurm-%j.out
#SBATCH --error=./slurm/slurm-%j.err
#SBATCH --nodes=1
#SBATCH --mem=32G

python gen_sr_dimacs.py datasets/development/train/ 500 --sat --unsat --min_n=10 --max_n=40

python gen_sr_dimacs.py datasets/development/validation 50 --sat --unsat --min_n=40 --max_n=40

python gen_sr_dimacs.py datasets/development/test 50 --sat --unsat --min_n=40 --max_n=40

