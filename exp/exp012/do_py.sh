#!/bin/bash
EXP_NAME=exp012
NOW=$(date +%Y%m%d%H%M%S)

uv run python exp/$EXP_NAME/gen_data.py --dir outputs/$EXP_NAME/$NOW/upload
gcloud storage cp -r outputs/$EXP_NAME/$NOW/upload gs://saito-map/outputs/$EXP_NAME/$NOW/upload

gcloud compute instances stop saito-gpu-map-dev --zone=us-central1-b --discard-local-ssd=false