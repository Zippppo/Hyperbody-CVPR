#!/bin/bash
#SBATCH --job-name=graph_dist
#SBATCH --partition=long
#SBATCH --nodelist=gpu15
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH --output=outputs/slurm_graph_dist_%j.log

source ~/.bashrc
conda activate pasco

cd /home/comp/csrkzhu/code/HyperBody-main

python scripts/precompute_graph_distance.py \
    --device cuda \
    --num-workers 0
