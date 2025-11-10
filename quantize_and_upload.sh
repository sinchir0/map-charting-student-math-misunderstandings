#!/bin/bash
gcloud auth login
uv run python exp/late_exp011/quantize.py
uv run python exp/late_exp012/quantize.py
uv run python exp/late_exp013/quantize.py
uv run python exp/late_exp011/data_upload.py --dir outputs/late_exp011_gptq --dataset-name late_exp011-gptq
uv run python exp/late_exp012/data_upload.py --dir outputs/late_exp012_awq --dataset-name late_exp012-awq
uv run python exp/late_exp013/data_upload.py --dir outputs/late_exp013/auto_round --dataset-name late_exp013-signround
gcloud compute instances stop saito-gpu-3 --zone=us-central1-c --discard-local-ssd=false