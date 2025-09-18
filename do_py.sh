#!/bin/bash
EXP_NAME=exp007
DIR_NAME=outputs/$EXP_NAME/$(date +%Y%m%d%H%M%S)
NOW=$(date +%Y%m%d%H%M%S)

uv run python exp/$EXP_NAME/train.py --dir $DIR_NAME
uv run python exp/$EXP_NAME/evaluate.py --dir $DIR_NAME/upload
uv run python exp/$EXP_NAME/data_upload.py --dir $DIR_NAME/upload --dataset-name $EXP_NAME-$NOW
aws s3 cp --recursive $DIR_NAME/upload s3://map-charting-student-math-misunderstandings/${DIR_NAME#outputs/}

gcloud compute instances stop saito-gpu-map-calc --zone=us-central1-a --discard-local-ssd=false