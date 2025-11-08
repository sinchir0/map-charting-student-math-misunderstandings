#!/bin/bash
EXP_NAME=late_exp004
NOW=$(TZ=Asia/Tokyo date +%Y%m%d%H%M%S)
DIR_NAME=outputs/$EXP_NAME/$NOW

gcloud auth login
uv run python exp/$EXP_NAME/train.py

# # Kaggleへアップロード
# uv run python exp/$EXP_NAME/data_upload.py --dir $DIR_NAME --dataset-name $EXP_NAME-$NOW

# # GCSへのアップロード
# gcloud storage cp -r $DIR_NAME gs://saito-map/${DIR_NAME#outputs/}

# インスタンスを落とす
gcloud compute instances stop saito-gpu --zone=us-central1-b --discard-local-ssd=false