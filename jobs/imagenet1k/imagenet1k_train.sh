#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=imagenet1k_train
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_train_%x_%j.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_train_%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=08:00:00
# NOT YET TIMED -- placeholder only, do not trust. 224x224 inputs on CIFAR-10/SVHN/MNIST-family/
# EuroSAT/Places365-scale data (full native training sets, up to 30 epochs with early stopping)
# have no timing precedent in this repo; TinyImageNet's 64x64 task-split numbers do not transfer.
# Before relying on this: time one target end-to-end, apply the standing 1.5x headroom rule, and
# update this number -- EuroSAT (smallest, ~22k train) and Places365-subset (largest) are the two
# most informative to time first, since they likely bracket the range.
#
# Usage: sbatch jobs/imagenet1k_train.sh <target>

TARGET="${1:?Usage: sbatch jobs/imagenet1k_train.sh <target>}"

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date) -- target=$TARGET"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m imagenet1k.imagenet1k_test --target "$TARGET" --arm both
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
