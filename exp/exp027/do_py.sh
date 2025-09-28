#!/bin/bash
EXP_NAME=exp024
NOW=$(TZ=Asia/Tokyo date +%Y%m%d%H%M%S)
# DIR_NAME=outputs/$EXP_NAME/$NOW
DIR_NAME=outputs/exp024/20250928124526

gcloud auth login
# uv run python exp/$EXP_NAME/train.py --dir $DIR_NAME

# checkpointを利用する場合、NOWの時間を既存のディレクトリに変更する
# uv run python exp/$EXP_NAME/train.py --dir $DIR_NAME --use_checkpoint

# DIR_NAME/checkpoint配下の全ディレクトリでevaluate.pyを実行
for d in $DIR_NAME/checkpoint/*/; do
    uv run python exp/$EXP_NAME/evaluate.py --dir "$d"
done

# validationの結果をまとめる
uv run python exp/$EXP_NAME/summarize_map_at_3.py --dir $DIR_NAME

# CVが最高のモデルを、uploadへコピーし、optimizer.ptを削除
uv run python exp/$EXP_NAME/copy_best_cv_to_upload.py --dir $DIR_NAME

# Kaggleへアップロード
uv run python exp/$EXP_NAME/data_upload.py --dir $DIR_NAME --dataset-name $EXP_NAME-$NOW

# GCSへのアップロード
gcloud storage cp -r $DIR_NAME gs://saito-map/${DIR_NAME#outputs/}

# インスタンスを落とす
gcloud compute instances stop saito-gpu-map-calc-1t --zone=us-central1-c --discard-local-ssd=false