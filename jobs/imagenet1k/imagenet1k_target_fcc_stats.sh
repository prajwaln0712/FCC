#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=imagenet1k_fcc_stats
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_target_fcc_stats_%x_%j.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_target_fcc_stats_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=09:00:00
# NOT YET TIMED -- placeholder only, do not trust (N=2000/class x num_classes images per target,
# through both Inception-768 and DINOv3-ViT-H+; the latter has no throughput precedent in this
# repo). Time one target, apply the standing 1.5x headroom rule, and update this number.
#
# Usage: sbatch jobs/imagenet1k_target_fcc_stats.sh <target>

TARGET="${1:?Usage: sbatch jobs/imagenet1k_target_fcc_stats.sh <target>}"

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date) -- target=$TARGET"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m imagenet1k.target_fcc_stats --target "$TARGET"
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
