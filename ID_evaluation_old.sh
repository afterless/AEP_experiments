#!/bin/bash
#SBATCH --job-name=ID_evaluation
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:H200:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=10G
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=apanda38@gatech.edu

uv run python ID_evaluation_old.py