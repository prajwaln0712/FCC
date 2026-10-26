#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --nodelist=skl-a-47
#SBATCH --job-name=verify_mnist
#SBATCH --output=/home/%u/logs/fcc/verify_mnist.out
#SBATCH --error=/home/%u/logs/fcc/verify_mnist.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=01:00:00

mkdir -p ~/logs/fcc

source ~/fcc_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
python -c "import torch, torchvision; print('torch', torch.__version__, '| torchvision', torchvision.__version__)"

echo "Downloading and verifying MNIST-family datasets (MNIST, KMNIST, FashionMNIST, USPS, EMNIST-letters)"
python verify_mnist_datasets.py
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
