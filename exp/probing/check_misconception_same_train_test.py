from pathlib import Path

import pandas as pd
from datetime import datetime
import pytz

import json
import random

DEBUG = False
COMPETITION_NAME = "map-charting-student-math-misunderstandings"
NOW = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d%H%M%S")
DATA_PATH = Path("data")
LR = 2e-5
EPOCH = 1
SEED = 42

def make_completion(df: pd.DataFrame) -> pd.DataFrame:
    df["Misconception"] = df["Misconception"].fillna("NA")
    df["completion"] = df["Category"] + ":" + df["Misconception"]
    n_classes = df["completion"].nunique()
    print(f"Train shape: {df.shape} with {n_classes} target classes")
    return df

def add_is_correct(df: pd.DataFrame) -> pd.DataFrame:
    """
    前提として、ラベル付けが誤っていることがある。
    よって、QuestionIdに対して、MC_AnswerがTrueになっている回答が最も多いものを、真の正解として扱う。
    """
    idx = df.apply(lambda row: row["Category"].split("_")[0], axis=1) == "True"
    correct = df.loc[idx].copy()
    correct["count"] = correct.groupby(["QuestionId", "MC_Answer"]).MC_Answer.transform(
        "count"
    )
    correct = correct.sort_values("count", ascending=False)
    correct = correct.drop_duplicates(["QuestionId"])
    correct = correct[["QuestionId", "MC_Answer"]]
    correct["is_correct"] = 1

    df = df.merge(correct, on=["QuestionId", "MC_Answer"], how="left")
    df["is_correct"] = df["is_correct"].fillna(0)
    return df

def calculate_map_at_k(y_true, y_pred, k=3):
    """
    Mean Average Precision at K を計算する
    y_true: 正解ラベル（文字列）のリスト
    y_pred: 予測ラベル（リスト）のリスト。各リストは上位K個の予測を含む
    k: 評価するランクの上限
    """
    average_precisions = []
    
    for true_label, pred_list in zip(y_true, y_pred):
        # 予測リストを上位k個に制限
        pred_k = pred_list[:k]
        
        # 正解が予測リストに含まれているかチェック
        if true_label in pred_k:
            # 正解の位置（1-indexed）
            position = pred_k.index(true_label) + 1
            # Average Precision = 1 / position
            ap = 1.0 / position
        else:
            ap = 0.0
        
        average_precisions.append(ap)
    
    # Mean Average Precision
    map_score = sum(average_precisions) / len(average_precisions)
    return map_score, average_precisions

if __name__ == "__main__":
    train = pd.read_csv(DATA_PATH / "train.csv")

    train = make_completion(train)

    all_completions = train["completion"].unique().tolist()

    # 誤解を含まないものを削除する
    all_completions

    # all_completionsからランダムに3つ選んだ候補を、trainの行数分作成する
    random.seed(42)
    predictions = []
    for _ in range(len(train)):
        preds = random.sample(all_completions, 3)
        predictions.append(preds)

    # MAP@3を計算
    map_at_3, aps = calculate_map_at_k(train["completion"].tolist(), predictions, k=3)
    print(f"RANDOM MAP@3: {map_at_3}")