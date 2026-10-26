#!/bin/bash
#SBATCH --account=rc-onboard
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --job-name=augment_1epoch
#SBATCH --output=/home/%u/logs/fcc/augment_1epoch.out
#SBATCH --error=/home/%u/logs/fcc/augment_1epoch.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00 

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

# TinyImageNet (data/tiny-ImageNet/tiny-imagenet-200) and CIFAR10 (data/CIFAR10) must already
# be on disk -- augment.py loads CIFAR10 with download=False. If DINOv3 weights aren't cached
# yet, drop the next line for this run so transformers can fetch them once, then re-add it.
export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch; print('torch', torch.__version__, '| cuda available', torch.cuda.is_available())"

echo "Running augment.py for 1 seed, 1 pretrain epoch, 1 adapt epoch (sizing run for the 300/30-epoch experiment)"
time python -m dataaugmentation.augment --seed 1 --epochs-pretrain 1 --epochs-adapt 1
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
echo "For peak memory and wall-clock, once this job's accounting lands, run: seff $SLURM_JOB_ID"
exit $STATUS
