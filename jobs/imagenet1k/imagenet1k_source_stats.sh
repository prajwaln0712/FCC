#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=imagenet1k_src_stats
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_source_stats.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_source_stats.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=15:00:00
# NOT YET TIMED -- placeholder only, do not trust. DINOv3-ViT-H+'s throughput embedding 500,000
# images has no precedent in this repo (only vits16/vitb16 have ever been run at scale here), and
# ViT-H+ is a substantially bigger/slower model. Before relying on this: run once with a small
# --n-images (e.g. 5000) to measure per-image throughput, extrapolate to 500000, apply the
# standing 1.5x headroom rule, and update this number.

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m imagenet1k.imagenet_source_stats --n-images "${N_IMAGES:-500000}"
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
