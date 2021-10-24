#!/usr/bin/env bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ "$PWD" != "$DIR" ]; then
    echo "Please run the script in the script's residing directory"
    exit 0
fi


model_config=$1
tag=$2

if [ "$tag" != "" ]; then
    out_dir="${model_config%.*}-$tag-logs"
else
    out_dir="${model_config%.*}-logs"
fi


python ../bevnet/train_single.py \
    --model_config="$model_config" \
    --dataset_config="../dataset_configs/kitti4_100x100_unknown_single.yaml" \
    --dataset_path="../data/semantic_kitti_4class_100x100/" \
    --output="$out_dir" \
    --batch_size=4 \
    --include_unknown \
    --log_interval=50 \
    "${@:3}"
