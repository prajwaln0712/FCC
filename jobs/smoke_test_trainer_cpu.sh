#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --nodelist=skl-a-47
#SBATCH --job-name=trainer_smoke_cpu
#SBATCH --output=/home/%u/logs/fcc/trainer_smoke_cpu.out
#SBATCH --error=/home/%u/logs/fcc/trainer_smoke_cpu.err
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
python -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available())"

echo "Running models/trainer.py smoke test on CPU-only node (no GPU partition -- exercises the full-precision fallback regardless of use_amp)"
python smoke_test_trainer.py
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
