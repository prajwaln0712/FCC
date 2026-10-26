#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --job-name=tiny_split_smoke
#SBATCH --output=/home/%u/logs/fcc/tiny_split_smoke.out
#SBATCH --error=/home/%u/logs/fcc/tiny_split_smoke.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=04:00:00

# CPU-only: tiny_split.py never touches the GPU, and a forward pass through an untrained
# ResNet18() for a shape check doesn't need one either. No --gres=gpu requested.
#
# TinyImageNet (data/tiny-ImageNet/tiny-imagenet-200) must already be on disk. This runs
# tiny_split.py for real (it only ever runs once -- main() has no partial-task mode), producing
# all 10 saved_split_task/data/trainset_{N}.pt / test_{N}.pt files (~66GB total), then checks
# those files and a real batch through the new models/resnet18.py.

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"

echo "Running tiny_split.py (produces the 10 task splits)..."
time python -m tinyimagenet.tiny_split
STATUS=$?
if [ $STATUS -ne 0 ]; then
    echo "tiny_split.py failed with exit code $STATUS"
    exit $STATUS
fi

echo "Running smoke_test_tiny_split.py (checks split integrity + ResNet18 shape)..."
time python -m tinyimagenet.smoke_test_tiny_split
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
