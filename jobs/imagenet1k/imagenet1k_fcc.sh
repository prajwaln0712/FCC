#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=tier3
#SBATCH --job-name=imagenet1k_fcc
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_fcc.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_fcc.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
# CPU-only: closed-form Frechet distance from cached stats (Stage 1 + Stage 3 outputs), no
# embedding or training here -- should be fast. 30 min is generous headroom, not yet measured.

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"

python -m imagenet1k.fcc_experiment
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
