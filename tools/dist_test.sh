#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3
PORT=${PORT:-29503}

if [ -z "$CONFIG" ] || [ -z "$CHECKPOINT" ] || [ -z "$GPUS" ]; then
    echo "Usage: $0 <config> <checkpoint> <gpus> [additional args]"
    echo "Example: $0 ./projects/configs/bevformer/bev_tiny_det_occ_apollo.py ./work_dirs/bev_tiny_det_occ_apollo/latest.pth 1"
    exit 1
fi

HAS_EVAL=0
for arg in "${@:4}"; do
    if [ "$arg" = "--eval" ]; then
        HAS_EVAL=1
        break
    fi
done

EVAL_ARGS=()
if [ $HAS_EVAL -eq 0 ]; then
    EVAL_ARGS=(--eval iou bbox)
fi

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} ${EVAL_ARGS[@]}

