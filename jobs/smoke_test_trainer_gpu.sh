#!/bin/bash
#SBATCH --account=rc-onboard
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --job-name=trainer_smoke_gpu
#SBATCH --output=/home/%u/logs/fcc/trainer_smoke_gpu.out
#SBATCH --error=/home/%u/logs/fcc/trainer_smoke_gpu.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:15:00

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available())"

echo "Running models/trainer.py smoke test on GPU (use_amp=True exercises real autocast/GradScaler path)"
python smoke_test_trainer.py
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
