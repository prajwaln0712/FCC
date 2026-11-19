#!/bin/bash
# Submits the full imagenet1k (ImageNet-1K FCC) pipeline:
#   Stage 0: verify_targets + dino_smoke (no dependency on each other)
#   Stage 1: imagenet_source_stats (depends on dino_smoke passing -- same DINOv3-ViT-H+ model,
#            no point burning a multi-hour GPU job if the smoke test already caught a problem;
#            this dependency is an implementation-time addition, not spelled out verbatim in the
#            approved plan's step list, but follows directly from the plan's own point that the
#            smoke test must pass before relying on the full run)
#   Stage 2: 7 imagenet1k_train jobs, one per target (no dependency)
#   Stage 3: 7 imagenet1k_target_fcc_stats jobs, one per target (depends on dino_smoke, same reason
#            as stage 1)
#   Stage 4: imagenet1k_fcc (depends on stage 1 + all of stage 3)
#   Stage 5: imagenet1k_analysis (depends on stage 4 + all of stage 2)
#
# Run this directly on the login node: bash jobs/imagenet1k/submit_all_imagenet1k.sh
set -euo pipefail

TARGETS=(cifar10 places365 eurosat svhn mnist fashionmnist kmnist)

echo "Submitting Stage 0: verify_targets + dino_smoke..."
VERIFY_JOBID=$(sbatch --parsable jobs/imagenet1k/imagenet1k_verify_targets.sh)
echo "  verify_targets -> job $VERIFY_JOBID"
SMOKE_JOBID=$(sbatch --parsable jobs/imagenet1k/imagenet1k_dino_smoke.sh)
echo "  dino_smoke -> job $SMOKE_JOBID"

echo "Submitting Stage 1: imagenet_source_stats (depends on dino_smoke)..."
SRC_JOBID=$(sbatch --parsable --dependency=afterok:$SMOKE_JOBID jobs/imagenet1k/imagenet1k_source_stats.sh)
echo "  imagenet_source_stats -> job $SRC_JOBID"

echo "Submitting Stage 2: 7 imagenet1k_train jobs (no dependency)..."
declare -A TRAIN_JOBID
for TARGET in "${TARGETS[@]}"; do
    JOBID=$(sbatch --parsable jobs/imagenet1k/imagenet1k_train.sh "$TARGET")
    TRAIN_JOBID[$TARGET]=$JOBID
    echo "  train $TARGET -> job $JOBID"
done

echo "Submitting Stage 3: 7 imagenet1k_target_fcc_stats jobs (depends on dino_smoke)..."
declare -A FCC_STATS_JOBID
for TARGET in "${TARGETS[@]}"; do
    JOBID=$(sbatch --parsable --dependency=afterok:$SMOKE_JOBID jobs/imagenet1k/imagenet1k_target_fcc_stats.sh "$TARGET")
    FCC_STATS_JOBID[$TARGET]=$JOBID
    echo "  target_fcc_stats $TARGET -> job $JOBID"
done

echo "Submitting Stage 4: imagenet1k_fcc (depends on imagenet_source_stats + all target_fcc_stats)..."
FCC_DEPS="afterok:$SRC_JOBID"
for TARGET in "${TARGETS[@]}"; do
    FCC_DEPS="${FCC_DEPS}:${FCC_STATS_JOBID[$TARGET]}"
done
FCC_JOBID=$(sbatch --parsable --dependency=$FCC_DEPS jobs/imagenet1k/imagenet1k_fcc.sh)
echo "  imagenet1k_fcc -> job $FCC_JOBID (depends on $FCC_DEPS)"

echo "Submitting Stage 5: imagenet1k_analysis (depends on imagenet1k_fcc + all train jobs)..."
ANALYSIS_DEPS="afterok:$FCC_JOBID"
for TARGET in "${TARGETS[@]}"; do
    ANALYSIS_DEPS="${ANALYSIS_DEPS}:${TRAIN_JOBID[$TARGET]}"
done
ANALYSIS_JOBID=$(sbatch --parsable --dependency=$ANALYSIS_DEPS jobs/imagenet1k/imagenet1k_analysis.sh)
echo "  imagenet1k_analysis -> job $ANALYSIS_JOBID (depends on $ANALYSIS_DEPS)"

echo ""
echo "All jobs submitted. Note: verify_targets ($VERIFY_JOBID) has no downstream dependency --"
echo "check its output manually before trusting the rest of the pipeline's results."
