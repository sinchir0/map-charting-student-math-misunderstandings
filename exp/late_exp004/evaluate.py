import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, LogitsProcessor
import torch
from vllm.lora.request import LoRARequest
import pandas as pd
from pathlib import Path

# ref: https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo
DATA_PATH = Path("data")
OUT_DIR = "outputs/late_exp004/20251108134354/upload"
MAX_LEN = 1024
SEED = 42
DEBUG = False
MODEL_NAME = "Qwen/Qwen3-0.6B"

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

if __name__ == "__main__":
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
        # quantization="bitsandbytes",
        # enable_lora=True,
    )
    # lora_req = LoRARequest("adapter", 1, str(OUT_DIR))

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
    # outputs = vllm_model.generate(prompts, sampling_params, lora_request=lora_req)
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
    
    # 結果をCSVに保存
    val_df[["prompt", "completion", "prediction", "score"]].to_csv(f"{OUT_DIR}/validation_results.csv", index=False)
    print(f"\n推論結果を {OUT_DIR}/validation_results.csv に保存しました")
