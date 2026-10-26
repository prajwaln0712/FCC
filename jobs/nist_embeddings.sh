#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=tier3
#SBATCH --gres=gpu:1
#SBATCH --job-name=nist_embeddings
#SBATCH --output=/home/%u/logs/fcc/nist_embeddings.out
#SBATCH --error=/home/%u/logs/fcc/nist_embeddings.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

# Weights must already be cached (download once on the login node); then uncomment to stay offline:
 export HF_HUB_OFFLINE=1

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -c "import torch, torchvision, transformers, pytorch_fid; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| available', torch.cuda.is_available(), '| torchvision', torchvision.__version__, '| transformers', transformers.__version__)"

echo "Extracting Inception + DINOv3 embeddings for the five *NIST datasets"
python nist_extract_embeddings.py --network "${NETWORK:-both}" --batch-size 256
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
