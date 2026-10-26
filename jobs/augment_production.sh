#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=augment_prod
#SBATCH --output=/home/%u/logs/fcc/augment_prod_%j.out
#SBATCH --error=/home/%u/logs/fcc/augment_prod_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=2-12:00:00

# Full 300-epoch Tiny-ImageNet pretrain / 30-epoch CIFAR10 adapt run, for one seed.
# Submit once per seed: sbatch jobs/augment_production.sh <seed>  (15 separate calls for 15 seeds)
export HF_HUB_OFFLINE=1

SEED="${1:-1}"

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available())"

echo "Running augment.py for seed $SEED, 300 pretrain epochs, 30 adapt epochs (production run)"
time python -m dataaugmentation.augment --seed "$SEED" --epochs-pretrain 300 --epochs-adapt 30
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
echo "For peak memory and wall-clock, once this job's accounting lands, run: seff $SLURM_JOB_ID"
exit $STATUS
