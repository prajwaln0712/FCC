#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --job-name=aug_scores
#SBATCH --output=/home/%u/logs/fcc/aug_scores_%x_%j.out
#SBATCH --error=/home/%u/logs/fcc/aug_scores_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00
# NOT YET MEASURED -- 1.5x an ESTIMATE, not a timing: estimated 15-30 min for 12 seeds (ResNet50
# forward over 50k 32x32 CIFAR-10 images per seed + LogME on 50k x 2048 features on CPU), 1.5 x 30 min
# = 45 min. Before trusting it: run once with --limit 2000 (prints scores + extract/score seconds,
# writes no CSV), scale the per-seed seconds up to 50k images, apply 1.5x, and update this number.
#
# Usage: sbatch jobs/other_metrics/aug_scores.sh [extra args, e.g. --seeds 1 2 or --limit 2000]

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m other_metrics.aug_scores "$@"
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
