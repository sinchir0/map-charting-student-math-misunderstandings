# ref: https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945

import os
import random
from pathlib import Path

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
import math

import os
import json

DEBUG = False
COMPETITION_NAME = "map-charting-student-math-misunderstandings"
NOW = datetime.now(pytz.timezone('Asia/Tokyo')).strftime("%Y%m%d%H%M%S")
EXP_NAME = "exp027_use_acemath"
MODEL_NAME = "nvidia/AceMath-7B-Instruct"
FOLD_PATH = Path("outputs/fold/stratified_folds.json")
DATA_PATH = Path("data")
ENV_PATH = Path("env_file")
MAX_LEN = 1156
BATCH_SIZE = 6
GRAD_ACCUM = 2
LR = 2e-5
EPOCH = 3
SEED = 42
PROMPT_FORMAT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}

Below are the available classifications you can choose from.
Always provide your response using only the specified format.

0: False_Correct:NA,
1: False_Misconception:Adding_across,
2: False_Misconception:Adding_terms,
3: False_Misconception:Additive,
4: False_Misconception:Base_rate,
5: False_Misconception:Certainty,
6: False_Misconception:Definition,
7: False_Misconception:Denominator-only_change,
8: False_Misconception:Division,
9: False_Misconception:Duplication,
a: False_Misconception:Firstterm,
b: False_Misconception:FlipChange,
c: False_Misconception:Ignores_zeroes,
d: False_Misconception:Incomplete,
e: False_Misconception:Incorrect_equivalent_fraction_addition,
f: False_Misconception:Interior,
g: False_Misconception:Inverse_operation,
h: False_Misconception:Inversion,
i: False_Misconception:Irrelevant,
j: False_Misconception:Longer_is_bigger,
k: False_Misconception:Mult,
l: False_Misconception:Multiplying_by_4,
m: False_Misconception:Not_variable,
n: False_Misconception:Positive,
o: False_Misconception:Scale,
p: False_Misconception:Shorter_is_bigger,
q: False_Misconception:Subtraction,
r: False_Misconception:SwapDividend,
s: False_Misconception:Tacking,
t: False_Misconception:Unknowable,
u: False_Misconception:WNB,
v: False_Misconception:Whole_numbers_larger,
w: False_Misconception:Wrong_Fraction,
x: False_Misconception:Wrong_Operation,
y: False_Misconception:Wrong_fraction,
z: False_Misconception:Wrong_term,
A: False_Neither:NA,
B: True_Correct:NA,
C: True_Misconception:Adding_across,
D: True_Misconception:Additive,
E: True_Misconception:Base_rate,
F: True_Misconception:Definition,
G: True_Misconception:Denominator-only_change,
H: True_Misconception:Division,
I: True_Misconception:Duplication,
J: True_Misconception:Firstterm,
K: True_Misconception:FlipChange,
L: True_Misconception:Incomplete,
M: True_Misconception:Incorrect_equivalent_fraction_addition,
N: True_Misconception:Inversion,
O: True_Misconception:Irrelevant,
P: True_Misconception:Longer_is_bigger,
Q: True_Misconception:Mult,
R: True_Misconception:Multiplying_by_4,
S: True_Misconception:Not_variable,
T: True_Misconception:Positive,
U: True_Misconception:Shorter_is_bigger,
V: True_Misconception:Subtraction,
W: True_Misconception:SwapDividend,
X: True_Misconception:Tacking,
Y: True_Misconception:WNB,
Z: True_Misconception:Whole_numbers_larger,
!: True_Misconception:Wrong_fraction,
#: True_Misconception:Wrong_term,
$: True_Neither:NA
"""
# NOTE: promptの最後に\を入れると、UserWarning: Mismatch between tokenized prompt and the start of tokenized prompt+completion. と表示される。

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

def change_completion_to_one_token(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "False_Correct:NA": "0",
        "False_Misconception:Adding_across": "1",
        "False_Misconception:Adding_terms": "2",
        "False_Misconception:Additive": "3",
        "False_Misconception:Base_rate": "4",
        "False_Misconception:Certainty": "5",
        "False_Misconception:Definition": "6",
        "False_Misconception:Denominator-only_change": "7",
        "False_Misconception:Division": "8",
        "False_Misconception:Duplication": "9",
        "False_Misconception:Firstterm": "a",
        "False_Misconception:FlipChange": "b",
        "False_Misconception:Ignores_zeroes": "c",
        "False_Misconception:Incomplete": "d",
        "False_Misconception:Incorrect_equivalent_fraction_addition": "e",
        "False_Misconception:Interior": "f",
        "False_Misconception:Inverse_operation": "g",
        "False_Misconception:Inversion": "h",
        "False_Misconception:Irrelevant": "i",
        "False_Misconception:Longer_is_bigger": "j",
        "False_Misconception:Mult": "k",
        "False_Misconception:Multiplying_by_4": "l",
        "False_Misconception:Not_variable": "m",
        "False_Misconception:Positive": "n",
        "False_Misconception:Scale": "o",
        "False_Misconception:Shorter_is_bigger": "p",
        "False_Misconception:Subtraction": "q",
        "False_Misconception:SwapDividend": "r",
        "False_Misconception:Tacking": "s",
        "False_Misconception:Unknowable": "t",
        "False_Misconception:WNB": "u",
        "False_Misconception:Whole_numbers_larger": "v",
        "False_Misconception:Wrong_Fraction": "w",
        "False_Misconception:Wrong_Operation": "x",
        "False_Misconception:Wrong_fraction": "y",
        "False_Misconception:Wrong_term": "z",
        "False_Neither:NA": "A",
        "True_Correct:NA": "B",
        "True_Misconception:Adding_across": "C",
        "True_Misconception:Additive": "D",
        "True_Misconception:Base_rate": "E",
        "True_Misconception:Definition": "F",
        "True_Misconception:Denominator-only_change": "G",
        "True_Misconception:Division": "H",
        "True_Misconception:Duplication": "I",
        "True_Misconception:Firstterm": "J",
        "True_Misconception:FlipChange": "K",
        "True_Misconception:Incomplete": "L",
        "True_Misconception:Incorrect_equivalent_fraction_addition": "M",
        "True_Misconception:Inversion": "N",
        "True_Misconception:Irrelevant": "O",
        "True_Misconception:Longer_is_bigger": "P",
        "True_Misconception:Mult": "Q",
        "True_Misconception:Multiplying_by_4": "R",
        "True_Misconception:Not_variable": "S",
        "True_Misconception:Positive": "T",
        "True_Misconception:Shorter_is_bigger": "U",
        "True_Misconception:Subtraction": "V",
        "True_Misconception:SwapDividend": "W",
        "True_Misconception:Tacking": "X",
        "True_Misconception:WNB": "Y",
        "True_Misconception:Whole_numbers_larger": "Z",
        "True_Misconception:Wrong_fraction": "!",
        "True_Misconception:Wrong_term": "#",
        "True_Neither:NA": "$"
    }

    df["completion"] = df["completion"].map(mapping)

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

    train = make_completion(train)
    train = change_completion_to_one_token(train)
    train = add_is_correct(train)
    train["prompt"] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train["prompt"].values[0])

    if DEBUG:
        train = train.sample(100, random_state=SEED).reset_index(drop=True)
        SAVE_STEPS = 0.5
        EVAL_STEPS = 0.5

    fold_dict = json.load(open(FOLD_PATH))
    train["fold"] = train["row_id"].astype(str).map(fold_dict)

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
    tokenizer.padding_side = "left"

    all_completions = train["completion"].unique().tolist()
    
    with open(f"{UPLOAD_PATH}/all_completions.json", "w", encoding="utf-8") as f:
        json.dump(all_completions, f, ensure_ascii=False, indent=2)
    print(f"Saved all_completions to {UPLOAD_PATH}/all_completions.json")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # lora_config = LoraConfig(
    #     r=8,
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     lora_alpha=64,
    #     lora_dropout=0.05,
    #     bias="none",
    #     task_type="CAUSAL_LM",
    # )

    optim_steps_per_epoch = math.ceil(len(train_ds) / (BATCH_SIZE * GRAD_ACCUM))
    total_optim_steps = optim_steps_per_epoch * EPOCH

    # 全体の10%間隔（= 全体を10分割）
    interval_steps = max(1, total_optim_steps // 10)

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
        logging_steps=interval_steps,
        save_steps=interval_steps,
        eval_steps=interval_steps,
        eval_strategy="steps",
        save_total_limit=6,
        bf16=True,
        tf32=True,
        fp16=False,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        report_to="wandb",
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        packing=False # A100なら動くかも
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
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
    # trainer.save_model(UPLOAD_PATH)
    tokenizer.save_pretrained(UPLOAD_PATH)