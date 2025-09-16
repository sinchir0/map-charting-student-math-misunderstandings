#!/bin/bash
uv run python exp/exp005/train.py
# uv run python exp/exp004/evaluate.py

gcloud compute instances stop saito-gpu-map-calc --zone=us-central1-a --discard-local-ssd=false