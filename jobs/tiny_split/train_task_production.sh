#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=train_task
#SBATCH --output=/home/%u/logs/fcc/train_task_%j.out
#SBATCH --error=/home/%u/logs/fcc/train_task_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:20:00

# Full 50-epoch run of train_task.py for one task. Submit once per task (0-9):
#   sbatch jobs/train_task_production.sh <task-id>
#
# Sizing from the 1-epoch smoke test (job 21958279): 10.69s/epoch x50 = ~534s (~8.9min).
# --time above gives ~2.2x headroom.

TASK_ID="${1:?Usage: sbatch train_task_production.sh <task-id>}"

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Running train_task.py for task $TASK_ID, 50 epochs"
time python -m tinyimagenet.train_task --task-id "$TASK_ID" --epochs 50
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
