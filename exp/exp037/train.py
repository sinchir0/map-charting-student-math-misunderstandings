# ref: https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945

import os
import random
from pathlib import Path
import math

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

DEBUG = False
COMPETITION_NAME = "map-charting-student-math-misunderstandings"
NOW = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d%H%M%S")
EXP_NAME = "exp037_add_all_choice_full_train"
MODEL_NAME = "Qwen/Qwen3-8B"
MISCONCEPTION_CANDIDATE_PATH = Path("outputs/question_id_to_misconception_candidate/question_id_to_misconception_candidate_half_label.json")
MISCONCEPTION_EXPLANATION_PATH = Path("outputs/misc_explanation/misconception.json")
DATA_PATH = Path("data/takaito_data")
ENV_PATH = Path("env_file")
MAX_LEN = 768
BATCH_SIZE = 8
GRAD_ACCUM = 2
SAVE_STEPS = 0.1
EVAL_STEPS = 0.1
LR = 2e-5
EPOCH = 3
SEED = 42
PROMPT_FORMAT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
All Choices: {AllChoice}
Correct Answer: {CorrectChoice}
Student's Answer: {MC_Answer}
Whether the student's answer is correct: {Correct}
Student Explanation: {StudentExplanation}
Candidates: {MisconceptionCandidate}

Using the information provided, determine whether the student has a correct understanding, a misconception, or neither.
If the student has a correct understanding, choose Correct.
If the student has a misconception, choose Misconception.
If you select Misconception, you must choose exactly one label from Candidates other than “Correct” or “Neither”.
If it is neither, choose Neither.

Below are the descriptions of each misconception.
{MisconceptionExplanation}

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
        AllChoice=row["AllChoice"], # NOTE: listのまま入れている。
        CorrectChoice=row["correct_choice"],
        MC_Answer=row["MC_Answer"],
        Correct="Yes" if row["is_correct"] else "No",
        StudentExplanation=row["StudentExplanation"],
        MisconceptionCandidate=row["misconception_candidate"],
        MisconceptionExplanation=row["misconception_explanation"],
    )

def add_is_correct_and_correct_choice(df: pd.DataFrame) -> pd.DataFrame:
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
    
    # 正解の選択肢を追加する
    df["correct_choice"] = df["QuestionId"].map(
        correct.set_index("QuestionId")["MC_Answer"]
    )
    return df

def make_str_label_to_symbol(df) -> dict[str, str]:
    tmp = df[["str_label","symbol_label"]].drop_duplicates()
    label2symbol_dict = dict(zip(tmp["str_label"], tmp["symbol_label"]))
    return label2symbol_dict

def add_all_choice(df: pd.DataFrame) -> pd.DataFrame:
    unique_df = df[["QuestionId", "MC_Answer"]].drop_duplicates()
    question_id_to_all_choice = unique_df.groupby("QuestionId")["MC_Answer"].apply(list)

    df["AllChoice"] = df["QuestionId"].map(question_id_to_all_choice)
    return df

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

    train = add_is_correct_and_correct_choice(train)
    train = add_all_choice(train)

    # misconception_candidate列を追加する
    with open(MISCONCEPTION_CANDIDATE_PATH, "r") as f:
        misconception_candidate_dict = json.load(f)
    label2symbol_dict = make_str_label_to_symbol(train)
    misconception_candidate_dict_symbol = misconception_candidate_dict.copy()
    for k, v in misconception_candidate_dict_symbol.items():
        misconception_candidate_dict_symbol[k] = ", ".join([label + f"|{label2symbol_dict[label]}" for label in v])
    train["misconception_candidate"] = train["QuestionId"].astype(str).map(misconception_candidate_dict_symbol)

    # misconception_explanation列を追加する
    with open(MISCONCEPTION_EXPLANATION_PATH, "r") as f:
        misconception_explanation_dict = json.load(f)
    question_id_to_explanation_text_dict = misconception_candidate_dict.copy()
    for k, v in question_id_to_explanation_text_dict.items():
        question_id_to_explanation_text_dict[k] = "\n".join([label + f":{misconception_explanation_dict[label]}" for label in v if label not in ["Correct", "Neither"]])
    train["misconception_explanation"] = train["QuestionId"].astype(str).map(question_id_to_explanation_text_dict)

    train["prompt"] = train.apply(format_input, axis=1)
    train["completion"] = train["symbol_label"]
    print("Example prompt for our LLM:")
    print(train["prompt"].values[0])
    
    if DEBUG:
        train = train.sample(100, random_state=SEED).reset_index(drop=True)
        SAVE_STEPS = 0.5
        EVAL_STEPS = 0.5

    train_df = train[train["fold"] != USE_FOLD].reset_index(drop=True)
    # val_df = train[train["fold"] == USE_FOLD].reset_index(drop=True)

    # val_df.to_csv(f"{UPLOAD_PATH}/val_df.csv", index=False)

    train_df = train.copy()

    train_ds = Dataset.from_pandas(train_df[COLS], preserve_index=False)
    # val_ds = Dataset.from_pandas(val_df[COLS], preserve_index=False)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2", # A100なら動くかも
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    # train_dsのpromptに対し、tokenizerの input_idsの長さがMAX_LENを超えていないことをチェックする、超えている場合はエラーを出力する
    def tokenize_fn(examples, tokenizer: PreTrainedTokenizer):
        # tokenizerのinput_idsの長さがMAX_LENを超えていないことをチェックする、超えている場合はエラーを出力する
        tokenized = tokenizer(
            examples["prompt"]
        )
        max_input_length = max([len(input_id) for input_id in tokenized["input_ids"]])
        if (max_input_length >= (MAX_LEN - 1)): # -1は、completionの分を確保するため
            raise ValueError(f"Input length exceeds MAX_LEN of {MAX_LEN - 1}. Input Length: {max_input_length}. Please increase MAX_LEN.")
        return tokenized
    
    # check
    _ = train_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=COLS)
    # _ = val_ds.map(lambda x: tokenize_fn(x, tokenizer), batched=True, remove_columns=COLS)

    all_completions = train["completion"].unique().tolist()
    
    with open(f"{UPLOAD_PATH}/all_completions.json", "w", encoding="utf-8") as f:
        json.dump(all_completions, f, ensure_ascii=False, indent=2)
    print(f"Saved all_completions to {UPLOAD_PATH}/all_completions.json")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    optim_steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_optim_steps = optim_steps_per_epoch * EPOCH

    # 全体の10%間隔（= 全体を10分割）
    interval_steps = max(1, total_optim_steps // 10)

    sft_config = SFTConfig(
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=BATCH_SIZE,
        # per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        num_train_epochs=EPOCH,
        max_length=MAX_LEN,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=interval_steps,
        save_steps=interval_steps,
        # eval_steps=interval_steps,
        # eval_strategy="steps",
        save_total_limit=6,
        bf16=True,
        tf32=True,
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="wandb",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        # packing=False # A100なら動くかも
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        # eval_dataset=val_ds,
        # peft_config=lora_config,
    )

    # CHECKPOINT_PATH の存在するパスのうち、最も数字が大きいものを選択する
    if args.use_checkpoint:
        checkpoint_dirs = [d for d in os.listdir(CHECKPOINT_PATH) if d.startswith("checkpoint-")]
        if checkpoint_dirs:
            latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))
            print(f"Resuming training from checkpoint: {latest_checkpoint}")
        else:
            print("No checkpoint found, starting training from scratch.")
            trainer.train()

    trainer.train(resume_from_checkpoint=f"{CHECKPOINT_PATH}/{latest_checkpoint}" if args.use_checkpoint else None)

    # 保存
    trainer.save_model(UPLOAD_PATH)
    tokenizer.save_pretrained(UPLOAD_PATH)