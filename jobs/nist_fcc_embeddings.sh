#!/bin/bash
#SBATCH --account=nlagent
#SBATCH --partition=tier3
#SBATCH --job-name=nist_fcc_emb
#SBATCH --output=/home/%u/logs/fcc/nist_fcc_embeddings.out
#SBATCH --error=/home/%u/logs/fcc/nist_fcc_embeddings.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --mail-user=slack:@pn2694
#SBATCH --mail-type=ALL
#SBATCH --time=03:00:00

mkdir -p ~/logs/fcc
source ~/ghn3_env/bin/activate
cd ~/fcc
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Job $SLURM_JOB_ID on $(hostname) started $(date)"
echo "Python: $(which python) ($(python --version 2>&1))"

FAILED=0
for NET in ${NETWORKS:-inception dino}; do
  MISSING=0
  for DS in MNIST FashionMNIST KMNIST USPS EMNIST-letters; do
    [ -f "nist_embeddings/${DS}_${NET}.pt" ] || { echo "missing nist_embeddings/${DS}_${NET}.pt"; MISSING=1; }
  done
  if [ $MISSING -eq 1 ]; then echo "Skipping $NET"; FAILED=1; continue; fi

  echo "=== FCC on $NET embeddings ($(date)) ==="
  python nist_experiment.py --features "$NET" --ridge "${RIDGE:-1e-6}" || FAILED=1
done

echo "Finished $(date) (any failure: $FAILED)"
exit $FAILED
