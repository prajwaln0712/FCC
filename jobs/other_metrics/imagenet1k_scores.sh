#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --job-name=imagenet1k_scores
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_scores_%x_%j.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_scores_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
# NOT YET MEASURED -- 1.5x an ESTIMATE, not a timing: estimated 15-30 min for all 7 targets
# (~400k images through ResNet18 at 224x224, limited by image loading/resizing, Places365 JPEGs
# slowest), 1.5 x 30 min = 45 min. Before trusting it: run once with --limit 2000 (prints scores +
# extract/score seconds per target, writes no CSV), scale up to each target's size, apply 1.5x, and
# update this number.
#
# Usage: sbatch jobs/other_metrics/imagenet1k_scores.sh [extra args, e.g. --targets cifar10 or --limit 2000]

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m other_metrics.imagenet1k_scores --workers "$SLURM_CPUS_PER_TASK" "$@"
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
