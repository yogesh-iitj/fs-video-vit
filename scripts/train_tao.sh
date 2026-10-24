#!/bin/bash
# Train + test on the TAO few-shot benchmark (tools/prepare_tao_fsvod.py) on one GPU:
#   0) smoke test of the full pipeline, 1) training, 2) test of the best checkpoint,
#   with temporal fusion disabled, and of the untrained OWLv2 baseline.
# Resumable: re-running continues from outputs/$RUN/last.pt and skips finished steps.
#
# Usage: bash scripts/train_tao.sh   (override with e.g. DATA=... EPOCHS=... PY=... bash scripts/train_tao.sh)
set -o pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${PY:-python}
DATA=${DATA:-data/tao_fsvod}
RUN=${RUN:-tao_owlv2L_5way5shot}
MODEL=${MODEL:-google/owlv2-large-patch14-ensemble}
EPOCHS=${EPOCHS:-20}
EPISODES=${EPISODES:-500}
WORKERS=${WORKERS:-12}
OUT=outputs/$RUN

cd $REPO || exit 1
mkdir -p $OUT logs
ulimit -n 65536 2>/dev/null || ulimit -n $(ulimit -Hn)

# 0) Smoke test of the full train + evaluate path (a few minutes)
if [ ! -f outputs/smoke/smoke_done ]; then
  SAMPLE=$(find -L $DATA/videos -name "*.jpg" | head -1)
  $PY tools/check_fused_attention.py $MODEL $SAMPLE 2>&1 | tee logs/smoke.log || { echo "fused attention check failed"; exit 1; }
  $PY train.py --data_dir $DATA --pretrained_model $MODEL --n_way 5 --k_shot 5 --max_frames 8 \
    --epochs 1 --episodes_per_epoch 4 --log_interval 1 --val_max_videos 2 --eval_max_frames 4 \
    --eval_before_training --skip_test --num_workers 4 --output_dir outputs --run_name smoke 2>&1 | tee -a logs/smoke.log \
  && $PY evaluate.py --data_dir $DATA --split test --checkpoint_path outputs/smoke/best_model.pt --max_videos 2 \
    --max_frames 4 --num_workers 4 --output_dir outputs/smoke/test --visualize --num_vis_episodes 1 --plot_results \
    2>&1 | tee -a logs/smoke.log \
  && touch outputs/smoke/smoke_done || { echo "smoke test failed"; exit 1; }
fi

# 1) Training (validation on 40 val videos x 16 frames after every epoch)
if [ ! -f $OUT/train_done ]; then
  $PY train.py --data_dir $DATA --pretrained_model $MODEL \
    --n_way 5 --k_shot 5 --max_frames 8 --epochs $EPOCHS --episodes_per_epoch $EPISODES \
    --val_max_videos 40 --eval_max_frames 16 --eval_before_training --skip_test \
    --num_workers $WORKERS --output_dir outputs --run_name $RUN 2>&1 | tee -a logs/$RUN.train.log \
    && touch $OUT/train_done || { echo "training failed"; exit 1; }
fi

# 2) Test: trained model, trained model without temporal fusion, untrained OWLv2 baseline
EVAL="$PY evaluate.py --data_dir $DATA --split test --n_way 5 --k_shot 5 --num_workers $WORKERS"
[ -f $OUT/test/evaluation_results.json ] || $EVAL --checkpoint_path $OUT/best_model.pt \
  --output_dir $OUT/test --visualize --num_vis_episodes 5 --plot_results 2>&1 | tee -a logs/$RUN.test.log
[ -f $OUT/test_no_temporal/evaluation_results.json ] || $EVAL --checkpoint_path $OUT/best_model.pt --no_temporal \
  --output_dir $OUT/test_no_temporal 2>&1 | tee -a logs/$RUN.test_no_temporal.log
[ -f $OUT/test_untrained/evaluation_results.json ] || $EVAL --pretrained_model $MODEL \
  --output_dir $OUT/test_untrained 2>&1 | tee -a logs/$RUN.test_untrained.log

grep -h -A3 "Overall results" logs/$RUN.test*.log
echo "all done"
