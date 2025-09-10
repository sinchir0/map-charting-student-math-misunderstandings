
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import pandas as pd
from pathlib import Path

# ref: https://www.kaggle.com/code/aerdem4/eedi-qwen32b-vllm-with-logits-processor-zoo
# ref: 
DATA_PATH = Path("data")
OUT_DIR = "outputs/exp001_cls/20250907233056"
MAX_LEN = 256
SEED = 42
DEBUG = True
if __name__ == "__main__":
    with open(f"{OUT_DIR}/all_completions.json", "r", encoding="utf-8") as f:
        all_completions = json.load(f)

    val_df = pd.read_csv(f"{OUT_DIR}/val_df.csv")
    if DEBUG:
        val_df = val_df[:10]
    tokenizer = AutoTokenizer.from_pretrained(OUT_DIR, trust_remote_code=True)

    allowed_token_ids = [tokenizer.encode(str(i), add_special_tokens=False)[0] for i in all_completions]

    vllm_model = LLM(
        model=OUT_DIR,  # 保存したモデルパスを指定
        trust_remote_code=True,
        dtype=torch.bfloat16,
        gpu_memory_utilization=0.95,
        max_model_len=MAX_LEN,
        seed=SEED,
    )

    # サンプリングパラメータ設定
    sampling_params = SamplingParams(
        temperature=0.0,  # greedy
        top_p=1,  # greedy
        top_k=-1,  # greedy
        max_tokens=1,
        logprobs=3, # MAP@3だから
        stop_token_ids=[tokenizer.eos_token_id],
        allowed_token_ids=allowed_token_ids
        # NOTE: MultipleChoiceLogitsProcessorは、vLLM のバージョン 0.10.1.1 では対応していない。
        # logits_processors=[
        #     MultipleChoiceLogitsProcessor(
        #         tokenizer,
        #         choices=all_completions
        #     )
        # ]
    )

    # val_df全体に対して推論実行
    print("\n=== vLLM推論開始 ===")
    prompts = val_df["prompt"].tolist()
    outputs = vllm_model.generate(prompts, sampling_params)

    predictions = []
    for output in outputs:
        prediction = [out.decoded_token for out in output.outputs[0].logprobs[0].values()]
        predictions.append(prediction)

    # 結果を保存
    val_df["prediction"] = predictions

    # 結果の一部を表示
    print(f"\nvLLM推論完了: {len(predictions)}件")
    for i in range(min(5, len(val_df))):
        print(f"\n--- Sample {i+1} ---")
        print(f"Ground Truth: {val_df.iloc[i]['completion']}")
        print(f"Prediction: {val_df.iloc[i]['prediction']}")

    # 結果をCSVに保存
    val_df[["prompt", "completion", "prediction"]].to_csv(f"{OUT_DIR}/validation_results.csv", index=False)
    print(f"\n推論結果を {OUT_DIR}/validation_results.csv に保存しました")

    # TODO: MAP@3の計算を追加する、計算方法から確認する