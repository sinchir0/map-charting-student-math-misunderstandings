#!/bin/bash
EXP_NAME=exp021
DIR_NAME=outputs/$EXP_NAME/$(TZ=Asia/Tokyo date +%Y%m%d%H%M%S)
NOW=$(TZ=Asia/Tokyo date +%Y%m%d%H%M%S)

gcloud auth login
uv run python exp/$EXP_NAME/train.py --dir $DIR_NAME

# DIR_NAME/checkpoint配下の全ディレクトリでevaluate.pyを実行
for d in $DIR_NAME/checkpoint/*/; do
    uv run python exp/$EXP_NAME/evaluate.py --dir "$d"
done

uv run python exp/$EXP_NAME/summarize_map_at_3.py --dir $DIR_NAME

gcloud storage cp -r $DIR_NAME gs://saito-map/${DIR_NAME#outputs/}
gcloud compute instances stop saito-gpu-map-calc-1t --zone=us-central1-c --discard-local-ssd=false