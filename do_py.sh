#!/bin/bash
uv run python exp/exp001/train.py
uv run python exp/exp001/evaluate.py

gcloud compute instances stop saito-gpu-map-dev --zone=us-central1-b