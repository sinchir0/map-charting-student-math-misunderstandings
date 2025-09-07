import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import torch
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from datasets import Dataset
import numpy as np
from transformers import AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
import torch, os

# ref: https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945

DATA_PATH = Path("data")
MAX_LEN = 256
OUT_DIR = "outputs/qwen3-0_6b-sft"
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 2e-5
EPOCH = 2
PROMPT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}
"""


os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(DATA_PATH / "train.csv")

le = LabelEncoder()

train["Misconception"] = train["Misconception"].fillna('NA')
train['target'] = train["Category"] + ":" + train["Misconception"]
# train['label'] = le.fit_transform(train['target'])
# target_classes = le.classes_
# n_classes = len(target_classes)
n_classes = train['target'].nunique()
print(f"Train shape: {train.shape} with {n_classes} target classes")

# Powerful Feature Engineer
# 前提として、ラベル付けが誤っていることがある。
# よって、QuestionIdに対して、MC_AnswerがTrueになっている回答が最も多いものを、真の正解として扱う。
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

def format_input(row) -> str:
    x = "Yes"
    if not row['is_correct']:
        x = "No"
    return PROMPT.format(
        QuestionText=row['QuestionText'],
        MC_Answer=row['MC_Answer'],
        Correct=x,
        StudentExplanation=row['StudentExplanation']
    )

train['text'] = train.apply(format_input, axis=1)
print("Example prompt for our LLM:")
print()
print( train.text.values[0] )

# Create 20% Validation Subset
# Split into train and validation sets
train_df, val_df = train_test_split(train, test_size=0.2, random_state=42)

# Convert to Hugging Face Dataset
# COLS = ['text','label'] 
COLS = ['text','target'] 
train_ds = Dataset.from_pandas(train_df[COLS], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[COLS], preserve_index=False)

# format for SFT
def preprocess_function(example):
    return {
        "prompt": example["text"],
        "completion": example['target'] # {"role": "assistant", "content": f"<think>{example['Complex_CoT']}</think>{example['Response']}"}
    }

train_ds = train_ds.map(preprocess_function, remove_columns=["text", "target"])
val_ds = val_ds.map(preprocess_function, remove_columns=["text", "target"])
pass
# TODO: 正しいフォーマットにできているか確認する

# === SFTTrainer + SFTConfig (L4/A100最適化、model_init_kwargs不使用) ===
# pip install -U trl accelerate transformers datasets
# 可能なら pip install flash-attn==2.* でFlashAttention2を有効化

# ---- GPU最適化（L4/A100想定）----
# torch.backends.cuda.matmul.allow_tf32 = True  # TF32（A100/L4可）
# try:
#     torch.set_float32_matmul_precision("high")  # 追加のTF32最適化
# except Exception:
#     pass

# tokenizer の pad 設定
# if tokenizer.pad_token is None:
#     tokenizer.pad_token = tokenizer.eos_token
# tokenizer.padding_side = "right"


# モデルを明示ロード（bf16）
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map="auto",
)

# 省メモリ: KVキャッシュOFFでの学習を許可（学習時は不要）
# if hasattr(model.config, "use_cache"):
#     model.config.use_cache = False

# ---- SFTConfig（学習設定）----
# 0.6B なのでL4/A100ならもう少しバッチを上げてもOKですが、安全に8x2で例示
sft_config = SFTConfig(
    output_dir=OUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LR,
    num_train_epochs=EPOCH,
    max_length=MAX_LEN,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=500,
    eval_steps=500,
    eval_strategy="steps",
    save_total_limit=2,
    bf16=True,  # L4/A100ならTrue
    tf32=True,  # A100/L4向け
    fp16=False,  # bf16優先
    gradient_checkpointing=True, # 省メモリ
    max_grad_norm=1.0,
    report_to="wandb",
    packing=True
)

# ---- SFTTrainer ----
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds
)

trainer.train()

# 保存
trainer.save_model(OUT_DIR)
# tokenizer.save_pretrained(OUT_DIR)

# ---- 簡易推論テスト ----
model.eval()
sample = val_df.iloc[0]["text"]
inputs = tokenizer(sample, return_tensors="pt").to(model.device)
with torch.no_grad():
    out = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
print(tokenizer.decode(out[0], skip_special_tokens=True))