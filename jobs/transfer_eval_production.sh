#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=transfer_eval
#SBATCH --output=/home/%u/logs/fcc/transfer_eval_%j.out
#SBATCH --error=/home/%u/logs/fcc/transfer_eval_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:15:00

# Full 20-epoch run of transfer_eval.py for one (source, target, load_pretrain) combination.
# Submit once per combination:
#   sbatch jobs/transfer_eval_production.sh <source> <target> <load_pretrain>
#
# load_pretrain=1 requires checkpoints/task_{source}/task_{source}_last.pt to already exist
# (i.e. that source's train_task_production.sh job must have completed first -- see
# jobs/submit_all_continual.sh, which submits these with the right Slurm --dependency).
#
# Sizing from the 1-epoch smoke test (job 21958279): worst case (real transfer) 8.37s/epoch x20
# = ~167s (~2.8min). --time above gives >5x headroom.

SOURCE="${1:?Usage: sbatch transfer_eval_production.sh <source> <target> <load_pretrain>}"
TARGET="${2:?Usage: sbatch transfer_eval_production.sh <source> <target> <load_pretrain>}"
LOAD_PRETRAIN="${3:?Usage: sbatch transfer_eval_production.sh <source> <target> <load_pretrain>}"

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "Running transfer_eval.py --source $SOURCE --target $TARGET --load_pretrain $LOAD_PRETRAIN, 20 epochs"
time python -m tinyimagenet.transfer_eval --source "$SOURCE" --target "$TARGET" --load_pretrain "$LOAD_PRETRAIN" --epochs 20
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
