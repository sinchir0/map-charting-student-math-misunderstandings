# ref: https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945

# TODO: 候補となる誤解と、そのラベルを整理する
# TODO: 学習が回るかを確認する。

import os
import random
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer
from datetime import datetime
import argparse
import json
import wandb
import pytz

import os
import json

DEBUG = True
COMPETITION_NAME = "map-charting-student-math-misunderstandings"
NOW = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d%H%M%S")
EXP_NAME = "exp032_use_takaito_data"
MODEL_NAME = "Qwen/Qwen3-8B"
MISCONCEPTION_CANDIDATE_PATH = Path("outputs/question_id_to_misconception_candidate/question_id_to_misconception_candidate_half_label.json")
LABEL2SYMBOL_PATH = Path("data/takaito_data/label2symbol_dict.pkl")
# FOLD_PATH = Path("outputs/fold/stratified_folds.json")
DATA_PATH = Path("data/takaito_data")
ENV_PATH = Path("env_file")
MAX_LEN = 512
BATCH_SIZE = 8
GRAD_ACCUM = 2
SAVE_STEPS = 0.1
EVAL_STEPS = 0.1
LR = 1e-4
EPOCH = 3
SEED = 42
PROMPT_FORMAT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
Answer: {MC_Answer}
Student Explanation: {StudentExplanation}
Candidates: {MisconceptionCandidate}

Using the information provided, determine whether the student has a correct understanding, a misconception, or neither.
If the student has a correct understanding, choose Correct.
If the student has a misconception, choose Misconception.
If you select Misconception, you must choose exactly one label from Candidates other than “Correct” or “Neither”.
If it is neither, choose Neither.

In the Candidates list, each label has a symbol appended after a vertical bar ( | ).
Generate with only the symbol of the label you choose.
"""

COLS = ["prompt", "completion"]
USE_FOLD = 0

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def format_input(row) -> str:
    return PROMPT_FORMAT.format(
        QuestionText=row["QuestionText"],
        MC_Answer=row["MC_Answer"],
        # Correct="Yes" if row["is_correct"] else "No",
        StudentExplanation=row["StudentExplanation"],
        MisconceptionCandidate=row["misconception_candidate"]
    )

# def change_completion_to_one_token(df: pd.DataFrame) -> pd.DataFrame:
#     mapping = {
#         "False_Correct:NA": "■",
#         "False_Misconception:Adding_across": "□",
#         "False_Misconception:Adding_terms": "▲",
#         "False_Misconception:Additive": "△",
#         "False_Misconception:Base_rate": "▼",
#         "False_Misconception:Certainty": "▽",
#         "False_Misconception:Definition": "◆",
#         "False_Misconception:Denominator-only_change": "◇",
#         "False_Misconception:Division": "○",
#         "False_Misconception:Duplication": "●",
#         "False_Misconception:Firstterm": "★",
#         "False_Misconception:FlipChange": "☆",
#         "False_Misconception:Ignores_zeroes": "♦",
#         "False_Misconception:Incomplete": "♥",
#         "False_Misconception:Incorrect_equivalent_fraction_addition": "♠",
#         "False_Misconception:Interior": "♣",
#         "False_Misconception:Inverse_operation": "§",
#         "False_Misconception:Inversion": "†",
#         "False_Misconception:Irrelevant": "‡",
#         "False_Misconception:Longer_is_bigger": "※",
#         "False_Misconception:Mult": "∞",
#         "False_Misconception:Multiplying_by_4": "±",
#         "False_Misconception:Not_variable": "≠",
#         "False_Misconception:Positive": "≈",
#         "False_Misconception:Scale": "√",
#         "False_Misconception:Shorter_is_bigger": "∑",
#         "False_Misconception:Subtraction": "∏",
#         "False_Misconception:SwapDividend": "∆",
#         "False_Misconception:Tacking": "Ω",
#         "False_Misconception:Unknowable": "μ",
#         "False_Misconception:WNB": "∂",
#         "False_Misconception:Whole_numbers_larger": "→",
#         "False_Misconception:Wrong_Fraction": "←",
#         "False_Misconception:Wrong_Operation": "↑",
#         "False_Misconception:Wrong_fraction": "↓",
#         "False_Misconception:Wrong_term": "↔",
#         "False_Neither:NA": "↕",
#         "True_Correct:NA": "〈",
#         "True_Misconception:Adding_across": "〉",
#         "True_Misconception:Additive": "『",
#         "True_Misconception:Base_rate": "』",
#         "True_Misconception:Definition": "│",
#         "True_Misconception:Denominator-only_change": "─",
#         "True_Misconception:Division": "┌",
#         "True_Misconception:Duplication": "┐",
#         "True_Misconception:Firstterm": "└",
#         "True_Misconception:FlipChange": "┘",
#         "True_Misconception:Incomplete": "┼",
#         "True_Misconception:Incorrect_equivalent_fraction_addition": "█",
#         "True_Misconception:Inversion": "▓",
#         "True_Misconception:Irrelevant": "▒",
#         "True_Misconception:Longer_is_bigger": "£",
#         "True_Misconception:Mult": "¥",
#         "True_Misconception:Multiplying_by_4": "€",
#         "True_Misconception:Not_variable": "₩",
#         "True_Misconception:Positive": "©",
#         "True_Misconception:Shorter_is_bigger": "®",
#         "True_Misconception:Subtraction": "™",
#         "True_Misconception:SwapDividend": "♪",
#         "True_Misconception:Tacking": "♫",
#         "True_Misconception:WNB": "☀",
#         "True_Misconception:Whole_numbers_larger": "☁",
#         "True_Misconception:Wrong_fraction": "☂",
#         "True_Misconception:Wrong_term": "☃",
#         "True_Neither:NA": "☎",
#     }

#     df["completion"] = df["completion"].map(mapping)

#     return df

def make_str_label_to_symbol(df) -> dict[str, str]:
    tmp = df[["str_label","symbol_label"]].drop_duplicates()
    label2symbol_dict = dict(zip(tmp["str_label"], tmp["symbol_label"]))
    return label2symbol_dict

if __name__ == "__main__":
    # Pathの指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='ディレクトリのパス')
    parser.add_argument('--use_checkpoint', action='store_const', const=True, default=None, help='チェックポイントを使用する場合は指定。指定しなければNone。')
    args = parser.parse_args()
    OUTPUT_PATH = args.dir if args.dir else f"outputs/{EXP_NAME}/{NOW}"
    CHECKPOINT_PATH = f"{OUTPUT_PATH}/checkpoint"
    UPLOAD_PATH = f"{OUTPUT_PATH}/upload"

    load_dotenv(f"{ENV_PATH}/.env")
    wandb.login(key=os.environ["WANDB_API_KEY"])
    wandb.init(project=COMPETITION_NAME, name=EXP_NAME)

    seed_everything(SEED)

    os.makedirs(UPLOAD_PATH, exist_ok=True)

    train = pd.read_csv(DATA_PATH / "train.csv")

    # train = make_completion(train)
    # train = change_completion_to_one_token(train)
    # train = add_is_correct(train)
    with open(MISCONCEPTION_CANDIDATE_PATH, "r") as f:
        misconception_candidate_dict = json.load(f)

    label2symbol_dict = make_str_label_to_symbol(train)

    # misconception_candidate_dict の valueに対して、:symbol を付与する
    for k, v in misconception_candidate_dict.items():
        misconception_candidate_dict[k] = " ,".join([label + f"|{label2symbol_dict[label]}" for label in v])

    train["misconception_candidate"] = train["QuestionId"].astype(str).map(misconception_candidate_dict)
    
    train["prompt"] = train.apply(format_input, axis=1)
    train["completion"] = train["symbol_label"]
    print("Example prompt for our LLM:")
    print(train["prompt"].values[0])
    pass
    if DEBUG:
        train = train.sample(100, random_state=SEED).reset_index(drop=True)
        SAVE_STEPS = 0.5
        EVAL_STEPS = 0.5

    train_df = train[train["fold"] != USE_FOLD].reset_index(drop=True)
    val_df = train[train["fold"] == USE_FOLD].reset_index(drop=True)

    val_df.to_csv(f"{UPLOAD_PATH}/val_df.csv", index=False)

    train_ds = Dataset.from_pandas(train_df[COLS], preserve_index=False)
    val_ds = Dataset.from_pandas(val_df[COLS], preserve_index=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", # A100なら動くかも
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    all_completions = train["completion"].unique().tolist()
    
    with open(f"{UPLOAD_PATH}/all_completions.json", "w", encoding="utf-8") as f:
        json.dump(all_completions, f, ensure_ascii=False, indent=2)
    print(f"Saved all_completions to {UPLOAD_PATH}/all_completions.json")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCH,
        max_length=MAX_LEN,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=0.1,
        save_steps=SAVE_STEPS,
        eval_steps=EVAL_STEPS,
        eval_strategy="steps",
        save_total_limit=10,
        bf16=True,
        tf32=True,
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="wandb",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        # packing=False # A100なら動くかも
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # peft_config=lora_config,
    )

    trainer.train(resume_from_checkpoint=CHECKPOINT_PATH if args.use_checkpoint else None)

    # 保存
    # trainer.save_model(UPLOAD_PATH)
    tokenizer.save_pretrained(UPLOAD_PATH)