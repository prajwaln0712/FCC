#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=fcc_score2
#SBATCH --output=/home/%u/logs/fcc/fcc_score2.out
#SBATCH --error=/home/%u/logs/fcc/fcc_score2.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=01:30:00

# FCC_score2: Inception-768 (block 2) + DINO-vitb16 (same model augment.py defaults to) across the
# 90 directed task pairs, with each class's real ~500 images padded to 800 via random
# flip/crop/color-jitter augmentation before embedding (keeps the original 768-d dimension, unlike
# compute_fcc_score1.py's lower-dimensional swap). Longer --time than fcc_score1.sh since padding
# adds ~300 extra images/class x20 classes x10 tasks x2 networks to embed.
#
# Requires saved_split_task/data/trainset_{0..9}.pt to already exist (from
# tiny_split_resnet18_smoke.sh).

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Running compute_fcc_score2.py..."
time python -m tinyimagenet.compute_fcc_score2
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
