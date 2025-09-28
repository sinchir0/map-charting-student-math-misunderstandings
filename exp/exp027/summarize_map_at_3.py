#!/usr/bin/env python3
import os
import json
import argparse
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, required=True, help='ベースディレクトリ (checkpointとupload配下がある)')
args = parser.parse_args()

base_dir = args.dir
checkpoint_dir = os.path.join(base_dir, "checkpoint")
upload_dir = os.path.join(base_dir, "upload")
out_file = os.path.join(upload_dir, "map_at_3_summary.txt")

results = []
if os.path.isdir(checkpoint_dir):
    def checkpoint_sort_key(x):
        if '-' in x and x.split('-')[-1].isdigit():
            return int(x.split('-')[-1])
        return float('inf')

    # checkpoint番号でソート
    dirs = sorted(os.listdir(checkpoint_dir), key=checkpoint_sort_key)
    for d in dirs:
        dir_path = os.path.join(checkpoint_dir, d)
        if not os.path.isdir(dir_path):
            continue
        eval_path = os.path.join(dir_path, "evaluation_results.json")
        if not os.path.isfile(eval_path):
            continue
        try:
            with open(eval_path, "r") as f:
                data = json.load(f)
            map_at_3 = data.get("map_at_3", None)
        except Exception as e:
            map_at_3 = None
        results.append([d, map_at_3])
else:
    print(f"checkpointディレクトリが見つかりません: {checkpoint_dir}")

os.makedirs(upload_dir, exist_ok=True)
with open(out_file.replace('.txt', '.csv'), "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["checkpoint_dir", "map_at_3"])
    writer.writerows(results)

print(f"Summary saved to {out_file.replace('.txt', '.csv')}")
