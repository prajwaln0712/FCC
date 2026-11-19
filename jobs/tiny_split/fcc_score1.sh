#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fcc_score1
#SBATCH --output=/home/%u/logs/fcc/fcc_score1.out
#SBATCH --error=/home/%u/logs/fcc/fcc_score1.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:45:00

# FCC_score1: Inception-192 + DINO-vits16 (no augmentation) across the 90 directed task pairs.
# Embeds each of the 10 tasks once per network (500 real images/class, no padding needed at these
# dimensions), caches to fcc_cache/score1/, then does closed-form Frechet-distance lookups for all
# 90 source != target pairs.
#
# Requires saved_split_task/data/trainset_{0..9}.pt to already exist (from
# tiny_split_resnet18_smoke.sh).

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Running compute_fcc_score1.py..."
time python -m tinyimagenet.compute_fcc_score1
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
