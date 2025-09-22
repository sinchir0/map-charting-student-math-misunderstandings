#!/bin/bash
EXP_NAME=exp013
DIR_NAME=outputs/$EXP_NAME/$(date +%Y%m%d%H%M%S)
NOW=$(date +%Y%m%d%H%M%S)

uv run python exp/$EXP_NAME/train.py --dir $DIR_NAME
uv run python exp/$EXP_NAME/evaluate.py --dir $DIR_NAME/upload
uv run python exp/$EXP_NAME/data_upload.py --dir $DIR_NAME/upload --dataset-name $EXP_NAME-$NOW
gcloud storage cp -r $DIR_NAME/upload gs://saito-map/${DIR_NAME#outputs/}

gcloud compute instances stop saito-gpu-map-calc --zone=us-central1-a --discard-local-ssd=false