#!/bin/bash
# Submits all 200 production jobs for the continual-learning experiment:
#   10 train_task.py runs (one per task, 50 epochs)
#   90 transfer_eval.py runs, --load_pretrain 1 (real transfer, source != target, 20 epochs)
#   90 transfer_eval.py runs, --load_pretrain 0 (floor baseline, source != target, 20 epochs)
#   10 transfer_eval.py runs, --load_pretrain 0, source == target (baselineA, 20 epochs)
#
# transfer_eval.py --load_pretrain 1 needs checkpoints/task_{source}/task_{source}_last.pt to
# already exist, so those submissions carry a Slurm --dependency=afterok:<train_task job id> on
# their source's train_task job -- no manual waiting or sleeping needed. The other 100 jobs
# (baseline + baselineA) don't load any checkpoint, so they have no dependency and can start
# immediately alongside the train_task jobs.
#
# Run this directly on the login node: bash jobs/submit_all_continual.sh
set -euo pipefail

declare -A TASK_JOBID

echo "Submitting 10 train_task.py jobs..."
for TASK in $(seq 0 9); do
    JOBID=$(sbatch --parsable jobs/train_task_production.sh "$TASK")
    TASK_JOBID[$TASK]=$JOBID
    echo "  task $TASK -> job $JOBID"
done

echo "Submitting 90 real-transfer (load_pretrain=1) jobs, each depending on its source's train_task job..."
for SOURCE in $(seq 0 9); do
    for TARGET in $(seq 0 9); do
        if [ "$SOURCE" -eq "$TARGET" ]; then
            continue
        fi
        JOBID=$(sbatch --parsable --dependency=afterok:${TASK_JOBID[$SOURCE]} jobs/transfer_eval_production.sh "$SOURCE" "$TARGET" 1)
        echo "  finetune source=$SOURCE target=$TARGET -> job $JOBID (depends on ${TASK_JOBID[$SOURCE]})"
    done
done

echo "Submitting 90 floor-baseline (load_pretrain=0, source!=target) jobs -- no dependency needed..."
for SOURCE in $(seq 0 9); do
    for TARGET in $(seq 0 9); do
        if [ "$SOURCE" -eq "$TARGET" ]; then
            continue
        fi
        JOBID=$(sbatch --parsable jobs/transfer_eval_production.sh "$SOURCE" "$TARGET" 0)
        echo "  baseline source=$SOURCE target=$TARGET -> job $JOBID"
    done
done

echo "Submitting 10 baselineA (load_pretrain=0, source==target) jobs..."
for TASK in $(seq 0 9); do
    JOBID=$(sbatch --parsable jobs/transfer_eval_production.sh "$TASK" "$TASK" 0)
    echo "  baselineA task $TASK -> job $JOBID"
done

echo "All 200 jobs submitted."
