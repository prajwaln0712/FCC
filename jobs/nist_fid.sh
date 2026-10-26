#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --nodelist=skl-a-47
#SBATCH --job-name=nist_fcc
#SBATCH --output=/home/%u/logs/fcc/nist_fcc.out
#SBATCH --error=/home/%u/logs/fcc/nist_fcc.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=02:00:00

mkdir -p ~/logs/fcc

source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"
python -c "import torch, torchvision, scipy; print('torch', torch.__version__, '| torchvision', torchvision.__version__, '| scipy', scipy.__version__)"

echo "Computing FCC on raw pixels for all 20 directed *NIST pairs"
python nist_experiment.py
STATUS=$?

echo "Finished $(date) with exit code $STATUS"
exit $STATUS
