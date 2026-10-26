#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=baselineB_eval
#SBATCH --output=/home/%u/logs/fcc/baselineB_eval.out
#SBATCH --error=/home/%u/logs/fcc/baselineB_eval.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00

# BaselineB: reads train_task.py's existing epoch-20 checkpoints (saved every 10 epochs during the
# 50-epoch Step 1 run) and evaluates each on its own task's test set -- inference only, no training.
# Replaces the old per-(source,target) baseline/ floor (frozen random backbone) with a genuine
# full-network-trained-on-target-alone baseline, one value per task (10 total), reused across all 9
# sources that target it.
#
# Requires checkpoints/task_{0..9}/task_{0..9}_20epoch.pt to already exist (from
# train_task_production.sh) and saved_split_task/data/test_{0..9}.pt to already exist (from
# tiny_split_resnet18_smoke.sh).

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Running baselineB evaluation (epoch-20 checkpoints, inference only)..."
time python -m tinyimagenet.eval_baselineB
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
