#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc_gpu
#SBATCH --job-name=cka_vit_b16
#SBATCH --output=/home/%u/logs/exp3/cka_vit_b16.out
#SBATCH --error=/home/%u/logs/exp3/cka_vit_b16.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00

source ~/ghn3_env/bin/activate
cd ~/ghn3

echo "Computing linear CKA (GHN-3-predicted vs ImageNet reference) for vit_b_16"

python compute_cka.py --model vit_b_16
