#!/bin/bash
gcloud auth login
bash exp/exp025/do_py.sh
bash exp/exp027/do_py.sh

gcloud compute instances stop saito-gpu-map-calc --zone=us-central1-a --discard-local-ssd=false