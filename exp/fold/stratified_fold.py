import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import json

DATA_PATH = Path("data")
SEED = 42
OUTPUT_PATH = Path("outputs/fold")

def make_completion(df: pd.DataFrame) -> pd.DataFrame:
    df["Misconception"] = df["Misconception"].fillna("NA")
    df["completion"] = df["Category"] + ":" + df["Misconception"]
    n_classes = df["completion"].nunique()
    print(f"Train shape: {df.shape} with {n_classes} target classes")
    return df

if __name__ == "__main__":
    train = pd.read_csv(DATA_PATH / "train.csv")
    train = make_completion(train)

    # StratifiedKFoldでfold番号を付与
    n_splits = 10
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    train["fold"] = -1
    for fold, (_, val_idx) in enumerate(skf.split(train, train["completion"])):
        train.loc[val_idx, "fold"] = fold

    print(train["fold"].value_counts().sort_index())
    print(train[train["fold"] == 0]["completion"].value_counts())
    print(train[train["fold"] == 1]["completion"].value_counts())
    
    row_id_fold_dict = dict(zip(train["row_id"], train["fold"]))

    # 保存
    with open(OUTPUT_PATH / "stratified_folds.json", "w", encoding="utf-8") as f:
        json.dump(row_id_fold_dict, f, ensure_ascii=False, indent=4)

    print(f"Saved {OUTPUT_PATH / 'stratified_folds.json'}")