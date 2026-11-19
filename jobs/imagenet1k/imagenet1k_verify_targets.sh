#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --job-name=imagenet1k_verify
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_verify_targets.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_verify_targets.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=05:00:00
# NOT YET TIMED: this is a first-run guess (first invocation also downloads Places365, EuroSAT,
# SVHN, CIFAR-10, MNIST-family). Re-time after the first real run and apply the standing 1.5x
# headroom rule before relying on this number again.

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"

python -m imagenet1k.verify_targets
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
