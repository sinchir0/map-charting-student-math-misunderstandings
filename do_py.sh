#!/bin/bash
uv run python exp/exp001_cls/train.py
uv run python exp/exp001_cls/evaluate.py

gcloud compute instances stop saito-gpu-map-dev --zone=us-central1-b