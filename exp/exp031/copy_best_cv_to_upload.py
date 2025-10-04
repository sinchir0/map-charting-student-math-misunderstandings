import argparse
import pandas as pd
import os

if __name__ == "__main__":
    # Pathの指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='ディレクトリのパス')
    args = parser.parse_args()
    SUMMARY_PATH = args.dir
    UPLOAD_PATH = os.path.join(SUMMARY_PATH, "upload")

    # SUMMARY_PATH / map_at_3_summary.csv に存在するファイルのmap_at_3が最も高いcheckpoint_dirを取得する
    # pandasを利用する
    df = pd.read_csv(os.path.join(SUMMARY_PATH, "upload", "map_at_3_summary.csv"))
    print("map_at_3_summary.csv:")
    print(df)
    best_row = df.loc[df["map_at_3"].idxmax()]
    best_checkpoint_dir = best_row["checkpoint_dir"]
    print(f"Best checkpoint_dir: {best_checkpoint_dir} with map_at_3: {best_row['map_at_3']}")

    # best_checkpoint_dir を upload にコピーし、optimizer.ptを削除する
    print(f"Copying best checkpoint files to {UPLOAD_PATH}...")
    src_path = os.path.join(SUMMARY_PATH, "checkpoint", best_checkpoint_dir)
    os.system(f"cp -r {src_path}/* {UPLOAD_PATH}/")
    optimizer_path = os.path.join(UPLOAD_PATH, "optimizer.pt")
    if os.path.exists(optimizer_path):
        os.remove(optimizer_path)
        print(f"Removed {optimizer_path}")
    else:
        print(f"No optimizer.pt found at {optimizer_path}")
    print(f"Copied files from {src_path} to {UPLOAD_PATH}")