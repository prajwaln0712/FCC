#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --job-name=continual_smoke
#SBATCH --output=/home/%u/logs/fcc/continual_smoke.out
#SBATCH --error=/home/%u/logs/fcc/continual_smoke.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00

# Sizing run for the real 50-epoch train_task.py / 20-epoch transfer_eval.py runs: 1 epoch of each
# of the four distinct training configurations, so we can measure per-epoch time/memory before
# requesting --time/--mem for the full 200-run sweep.
#
# Requires saved_split_task/data/trainset_{0,1}.pt / test_{0,1}.pt to already exist (produced by
# jobs/tiny_split_resnet18_smoke.sh).

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available())"

run_step() {
    DESC="$1"
    shift
    echo "=== $DESC ==="
    time python -m "$@"
    STATUS=$?
    if [ $STATUS -ne 0 ]; then
        echo "$DESC failed with exit code $STATUS"
        exit $STATUS
    fi
}

run_step "[1/4] train_task.py --task-id 0 --epochs 1 (sizes Step 1's per-epoch cost; first real exercise of models/resnet18.py)" \
    tinyimagenet.train_task --task-id 0 --epochs 50

run_step "[2/4] transfer_eval.py --source 0 --target 1 --load_pretrain 1 --epochs 1 (sizes real transfer)" \
    tinyimagenet.transfer_eval --source 0 --target 1 --load_pretrain 1 --epochs 20

run_step "[3/4] transfer_eval.py --source 0 --target 1 --load_pretrain 0 --epochs 1 (sizes the floor baseline)" \
    tinyimagenet.transfer_eval --source 0 --target 1 --load_pretrain 0 --epochs 20

run_step "[4/4] transfer_eval.py --source 0 --target 0 --load_pretrain 0 --epochs 1 (sizes baselineA)" \
    tinyimagenet.transfer_eval --source 0 --target 0 --load_pretrain 0 --epochs 20

echo "Finished $(date) with exit code 0"
echo "For peak memory and wall-clock, once this job's accounting lands, run: seff $SLURM_JOB_ID"
exit 0
