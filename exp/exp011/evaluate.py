import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, LogitsProcessor
import torch
import pandas as pd
from pathlib import Path
import argparse

# ref: https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo
DATA_PATH = Path("data")
# OUT_DIR = "outputs/exp004_8b_ep2/20250915145616/upload"
# MAX_LEN = 256
MAX_LEN = 1536
SEED = 42
DEBUG = False

os.environ["VLLM_USE_V1"] = "0" # Kaggle環境に合わせるため

# MAP@3の計算
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

# https://www.kaggle.com/code/aleaiest/lb-0-945-qwen2-5-32b-gptq/notebook
class LabelOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        mask = torch.full_like(scores, float('-inf'))
        if scores.dim() == 1:
            mask[self.allowed_token_ids] = 0
        elif scores.dim() == 2:
            mask[:, self.allowed_token_ids] = 0
        else:
            raise ValueError("Unexpected score dimensions")
        return scores + mask

def change_to_original_labels(val_df):
    inverted_mapping = {
        "■": "False_Correct:NA",
        "□": "False_Misconception:Adding_across",
        "▲": "False_Misconception:Adding_terms",
        "△": "False_Misconception:Additive",
        "▼": "False_Misconception:Base_rate",
        "▽": "False_Misconception:Certainty",
        "◆": "False_Misconception:Definition",
        "◇": "False_Misconception:Denominator-only_change",
        "○": "False_Misconception:Division",
        "●": "False_Misconception:Duplication",
        "★": "False_Misconception:Firstterm",
        "☆": "False_Misconception:FlipChange",
        "♦": "False_Misconception:Ignores_zeroes",
        "♥": "False_Misconception:Incomplete",
        "♠": "False_Misconception:Incorrect_equivalent_fraction_addition",
        "♣": "False_Misconception:Interior",
        "§": "False_Misconception:Inverse_operation",
        "†": "False_Misconception:Inversion",
        "‡": "False_Misconception:Irrelevant",
        "※": "False_Misconception:Longer_is_bigger",
        "∞": "False_Misconception:Mult",
        "±": "False_Misconception:Multiplying_by_4",
        "≠": "False_Misconception:Not_variable",
        "≈": "False_Misconception:Positive",
        "√": "False_Misconception:Scale",
        "∑": "False_Misconception:Shorter_is_bigger",
        "∏": "False_Misconception:Subtraction",
        "∆": "False_Misconception:SwapDividend",
        "Ω": "False_Misconception:Tacking",
        "μ": "False_Misconception:Unknowable",
        "∂": "False_Misconception:WNB",
        "→": "False_Misconception:Whole_numbers_larger",
        "←": "False_Misconception:Wrong_Fraction",
        "↑": "False_Misconception:Wrong_Operation",
        "↓": "False_Misconception:Wrong_fraction",
        "↔": "False_Misconception:Wrong_term",
        "↕": "False_Neither:NA",
        "〈": "True_Correct:NA",
        "〉": "True_Misconception:Adding_across",
        "『": "True_Misconception:Additive",
        "』": "True_Misconception:Base_rate",
        "│": "True_Misconception:Definition",
        "─": "True_Misconception:Denominator-only_change",
        "┌": "True_Misconception:Division",
        "┐": "True_Misconception:Duplication",
        "└": "True_Misconception:Firstterm",
        "┘": "True_Misconception:FlipChange",
        "┼": "True_Misconception:Incomplete",
        "█": "True_Misconception:Incorrect_equivalent_fraction_addition",
        "▓": "True_Misconception:Inversion",
        "▒": "True_Misconception:Irrelevant",
        "£": "True_Misconception:Longer_is_bigger",
        "¥": "True_Misconception:Mult",
        "€": "True_Misconception:Multiplying_by_4",
        "₩": "True_Misconception:Not_variable",
        "©": "True_Misconception:Positive",
        "®": "True_Misconception:Shorter_is_bigger",
        "™": "True_Misconception:Subtraction",
        "♪": "True_Misconception:SwapDividend",
        "♫": "True_Misconception:Tacking",
        "☀": "True_Misconception:WNB",
        "☁": "True_Misconception:Whole_numbers_larger",
        "☂": "True_Misconception:Wrong_fraction",
        "☃": "True_Misconception:Wrong_term",
        "☎": "True_Neither:NA",
    }

    ground_truth = [
        inverted_mapping[completion] for completion in val_df["completion"].tolist()
    ]

    val_df["completion"] = ground_truth

    prediction = [
        [inverted_mapping[pred] for pred in prediction]
        for prediction in val_df["prediction"].tolist()
    ]

    val_df["prediction"] = prediction

    return val_df

if __name__ == "__main__":
    # Pathの指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help='ディレクトリのパス')
    args = parser.parse_args()
    OUT_DIR = args.dir

    with open(f"{OUT_DIR}/all_completions.json", "r", encoding="utf-8") as f:
        all_completions = json.load(f)

    val_df = pd.read_csv(f"{OUT_DIR}/val_df.csv")
    if DEBUG:
        val_df = val_df[:10]
    
    tokenizer = AutoTokenizer.from_pretrained(OUT_DIR, trust_remote_code=True)

    allowed_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in all_completions]

    vllm_model = LLM(
        model=str(OUT_DIR),
        dtype=torch.float16, # Kaggle環境ではbfloat16が使えないため、合わせる
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        max_model_len=MAX_LEN,
        seed=SEED,
    )

    # サンプリングパラメータ設定
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy
        top_p=1,  # greedy
        top_k=-1,  # greedy
        max_tokens=1,
        logprobs=3, # because MAP@3
        stop_token_ids=[tokenizer.eos_token_id],
        logits_processors=[LabelOnlyLogitsProcessor(allowed_token_ids)],
    )

    # val_df全体に対して推論実行
    print("\n=== vLLM推論開始 ===")
    prompts = val_df["prompt"].tolist()
    outputs = vllm_model.generate(prompts, sampling_params)

    predictions = []
    for output in outputs:
        predicted_token_ids = list(output.outputs[0].logprobs[0])
        prediction = [tokenizer.decode([predicted_token_id]) for predicted_token_id in predicted_token_ids]
        predictions.append(prediction)

    # 結果を保存
    val_df["prediction"] = predictions

    # 結果の一部を表示
    print(f"\nvLLM推論完了: {len(predictions)}件")
    for i in range(min(5, len(val_df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Ground Truth: {val_df.iloc[i]['completion']}")
        print(f"Prediction: {val_df.iloc[i]['prediction']}")

    # MAP@3を計算
    map_at_3, aps = calculate_map_at_k(val_df["completion"].tolist(), val_df["prediction"].tolist(), k=3)
    
    # val_dfにscore列を追加
    val_df["score"] = aps
    
    print(f"\n=== 評価結果 ===")
    print(f"MAP@3: {map_at_3:.4f}")
    
    # 詳細な結果を表示
    print(f"\n=== 詳細結果（先頭5件） ===")
    for i in range(min(5, len(val_df))):
        print(f"Sample {i+1}:")
        print(f"  Ground Truth: {val_df.iloc[i]['completion']}")
        print(f"  Predictions: {val_df.iloc[i]['prediction']}")
        print(f"  Score: {val_df.iloc[i]['score']:.4f}")
    
    # 結果をJSONファイルに保存
    results = {
        "map_at_3": map_at_3,
        "num_samples": len(val_df),
        "debug_mode": DEBUG,
        "model_path": OUT_DIR
    }
    
    with open(f"{OUT_DIR}/evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n評価結果を {OUT_DIR}/evaluation_results.json に保存しました")
    
    # iconを元のラベルに戻す
    val_df = change_to_original_labels(val_df)

    # row_idを小さい順にソートする
    val_df = val_df.sort_values("row_id")

    # 結果をCSVに保存
    val_df[["row_id", "prompt", "completion", "prediction", "score"]].to_csv(f"{OUT_DIR}/validation_results.csv", index=False)
    print(f"\n推論結果を {OUT_DIR}/validation_results.csv に保存しました")
