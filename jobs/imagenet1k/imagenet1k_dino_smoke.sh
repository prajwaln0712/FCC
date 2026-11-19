#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=sporc-gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=imagenet1k_dino_smoke
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_dino_smoke.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_dino_smoke.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:30:00
# Must run (and pass) before imagenet1k_source_stats.sh / imagenet1k_target_fcc_stats.sh --
# resolves the pending DINOv3-ViT-H+ end-to-end verification item (only its config has been
# confirmed so far, not an actual forward pass). Only embeds 8 images, so 30 min is already
# generous headroom over the expected runtime (mostly model download/load time).

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv

python -m imagenet1k.smoke_test_dino_vith16plus
STATUS=$?
echo "Finished $(date) status=$STATUS"
exit $STATUS
