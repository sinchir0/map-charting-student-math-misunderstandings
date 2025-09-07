# ref: https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945

import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

import wandb

COMPETITION_NAME = "map-charting-student-math-misunderstandings"
EXP_NAME = "exp001"
DATA_PATH = Path("data")
ENV_PATH = Path("env_file")
MAX_LEN = 256
OUT_DIR = f"outputs/{EXP_NAME}"
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 2e-5
EPOCH = 1
SEED = 42
MODEL_NAME = "Qwen/Qwen3-0.6B"
DEBUG = True
PROMPT_FORMAT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}
"""
COLS = ["prompt", "completion"]


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def format_input(row) -> str:
    return PROMPT_FORMAT.format(
        QuestionText=row["QuestionText"],
        MC_Answer=row["MC_Answer"],
        Correct="Yes" if row["is_correct"] else "No",
        StudentExplanation=row["StudentExplanation"],
    )

load_dotenv(f"{ENV_PATH}/.env")
wandb.login(key=os.environ["WANDB_API_KEY"])
wandb.init(project=COMPETITION_NAME, name=EXP_NAME)

seed_everything(SEED)

os.makedirs(OUT_DIR, exist_ok=True)

train = pd.read_csv(DATA_PATH / "train.csv")

train = make_completion(train)
train = add_is_correct(train)
train["prompt"] = train.apply(format_input, axis=1)
print("Example prompt for our LLM:")
print()
print(train["prompt"].values[0])

if DEBUG:
    train = train.sample(1000, random_state=SEED).reset_index(drop=True)

train_df, val_df = train_test_split(train, test_size=0.1, random_state=42)

train_ds = Dataset.from_pandas(train_df[COLS], preserve_index=False)
val_ds = Dataset.from_pandas(val_df[COLS], preserve_index=False)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    dtype=torch.bfloat16,
    # attn_implementation="flash_attention_2", # A100なら動くかも
    device_map="auto",
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

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
    logging_steps=0.1,
    save_steps=0.1,
    eval_steps=0.1,
    eval_strategy="steps",
    save_total_limit=2,
    bf16=True,
    tf32=True,
    fp16=False,
    gradient_checkpointing=True,
    max_grad_norm=1.0,
    report_to="wandb",
    # packing=True # A100なら動くかも
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=sft_config,
    train_dataset=train_ds,
    eval_dataset=val_ds,
)

trainer.train()

# 保存
trainer.save_model(OUT_DIR)

# ---- 簡易推論テスト ----
model.eval()
sample = val_df.iloc[50]["prompt"]
answer = val_df.iloc[50]["completion"]
print(sample)
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
# 入力長を取得し、生成部分だけdecode
input_length = inputs["input_ids"].shape[1]
generated_tokens = out[0][input_length:]
print(tokenizer.decode(generated_tokens, skip_special_tokens=True))
