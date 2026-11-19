#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=debug
#SBATCH --job-name=imagenet1k_analysis
#SBATCH --output=/home/%u/logs/fcc/imagenet1k_analysis.out
#SBATCH --error=/home/%u/logs/fcc/imagenet1k_analysis.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=00:15:00
# CPU-only, short: merges Stage 2 + Stage 4 outputs into imagenet1k_results.csv and the
# correlation table/plots. 15 min is generous headroom, not yet measured.

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"

python -m imagenet1k.imagenet1k_analysis
STATUS=$?
if [ $STATUS -eq 0 ]; then
  python -c "import plots; plots.plot_all_imagenet1k()"
  STATUS=$?
fi
echo "Finished $(date) status=$STATUS"
exit $STATUS
