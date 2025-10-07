#!/bin/bash
gcloud auth login

bash exp/exp032/do_py.sh
bash exp/exp033/do_py.sh
bash exp/exp035/do_py.sh

gcloud compute instances stop saito-gpu-map-calc-1t --zone=us-central1-c --discard-local-ssd=false