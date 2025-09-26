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

import os
import json


COMPETITION_NAME = "map-charting-student-math-misunderstandings"
NOW = datetime.now().strftime("%Y%m%d%H%M%S")
EXP_NAME = "exp016_add_choice_and_correct_answer"
MODEL_NAME = "Qwen/Qwen3-8B"
FOLD_PATH = Path("outputs/fold/stratified_folds.json")
DATA_PATH = Path("data")
ENV_PATH = Path("env_file")
# MAX_LEN = 256
# MAX_LEN = 1024
MAX_LEN = 1152
# BATCH_SIZE = 8
BATCH_SIZE = 6
GRAD_ACCUM = 2
LR = 2e-5
EPOCH = 3
SEED = 42
PROMPT_FORMAT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
All Choices: {AllChoice}
Correct Answer: {CorrectChoice}
Student's Choice: {MC_Answer}
Is Correct?: {Correct}
Student Explanation: {StudentExplanation}

Below are the available classifications you can choose from.
Always provide your response using only the specified format.

■: False_Correct:NA,
□: False_Misconception:Adding_across,
▲: False_Misconception:Adding_terms,
△: False_Misconception:Additive,
▼: False_Misconception:Base_rate,
▽: False_Misconception:Certainty,
◆: False_Misconception:Definition,
◇: False_Misconception:Denominator-only_change,
○: False_Misconception:Division,
●: False_Misconception:Duplication,
★: False_Misconception:Firstterm,
☆: False_Misconception:FlipChange,
♦: False_Misconception:Ignores_zeroes,
♥: False_Misconception:Incomplete,
♠: False_Misconception:Incorrect_equivalent_fraction_addition,
♣: False_Misconception:Interior,
§: False_Misconception:Inverse_operation,
†: False_Misconception:Inversion,
‡: False_Misconception:Irrelevant,
※: False_Misconception:Longer_is_bigger,
∞: False_Misconception:Mult,
±: False_Misconception:Multiplying_by_4,
≠: False_Misconception:Not_variable,
≈: False_Misconception:Positive,
√: False_Misconception:Scale,
∑: False_Misconception:Shorter_is_bigger,
∏: False_Misconception:Subtraction,
∆: False_Misconception:SwapDividend,
Ω: False_Misconception:Tacking,
μ: False_Misconception:Unknowable,
∂: False_Misconception:WNB,
→: False_Misconception:Whole_numbers_larger,
←: False_Misconception:Wrong_Fraction,
↑: False_Misconception:Wrong_Operation,
↓: False_Misconception:Wrong_fraction,
↔: False_Misconception:Wrong_term,
↕: False_Neither:NA,
〈: True_Correct:NA,
〉: True_Misconception:Adding_across,
『: True_Misconception:Additive,
』: True_Misconception:Base_rate,
│: True_Misconception:Definition,
─: True_Misconception:Denominator-only_change,
┌: True_Misconception:Division,
┐: True_Misconception:Duplication,
└: True_Misconception:Firstterm,
┘: True_Misconception:FlipChange,
┼: True_Misconception:Incomplete,
█: True_Misconception:Incorrect_equivalent_fraction_addition,
▓: True_Misconception:Inversion,
▒: True_Misconception:Irrelevant,
£: True_Misconception:Longer_is_bigger,
¥: True_Misconception:Mult,
€: True_Misconception:Multiplying_by_4,
₩: True_Misconception:Not_variable,
©: True_Misconception:Positive,
®: True_Misconception:Shorter_is_bigger,
™: True_Misconception:Subtraction,
♪: True_Misconception:SwapDividend,
♫: True_Misconception:Tacking,
☀: True_Misconception:WNB,
☁: True_Misconception:Whole_numbers_larger,
☂: True_Misconception:Wrong_fraction,
☃: True_Misconception:Wrong_term,
☎: True_Neither:NA
"""
COLS = ["prompt", "completion"]
DEBUG = False
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

def change_completion_to_one_token(df: pd.DataFrame) -> pd.DataFrame:
    mapping = {
        "False_Correct:NA": "■",
        "False_Misconception:Adding_across": "□",
        "False_Misconception:Adding_terms": "▲",
        "False_Misconception:Additive": "△",
        "False_Misconception:Base_rate": "▼",
        "False_Misconception:Certainty": "▽",
        "False_Misconception:Definition": "◆",
        "False_Misconception:Denominator-only_change": "◇",
        "False_Misconception:Division": "○",
        "False_Misconception:Duplication": "●",
        "False_Misconception:Firstterm": "★",
        "False_Misconception:FlipChange": "☆",
        "False_Misconception:Ignores_zeroes": "♦",
        "False_Misconception:Incomplete": "♥",
        "False_Misconception:Incorrect_equivalent_fraction_addition": "♠",
        "False_Misconception:Interior": "♣",
        "False_Misconception:Inverse_operation": "§",
        "False_Misconception:Inversion": "†",
        "False_Misconception:Irrelevant": "‡",
        "False_Misconception:Longer_is_bigger": "※",
        "False_Misconception:Mult": "∞",
        "False_Misconception:Multiplying_by_4": "±",
        "False_Misconception:Not_variable": "≠",
        "False_Misconception:Positive": "≈",
        "False_Misconception:Scale": "√",
        "False_Misconception:Shorter_is_bigger": "∑",
        "False_Misconception:Subtraction": "∏",
        "False_Misconception:SwapDividend": "∆",
        "False_Misconception:Tacking": "Ω",
        "False_Misconception:Unknowable": "μ",
        "False_Misconception:WNB": "∂",
        "False_Misconception:Whole_numbers_larger": "→",
        "False_Misconception:Wrong_Fraction": "←",
        "False_Misconception:Wrong_Operation": "↑",
        "False_Misconception:Wrong_fraction": "↓",
        "False_Misconception:Wrong_term": "↔",
        "False_Neither:NA": "↕",
        "True_Correct:NA": "〈",
        "True_Misconception:Adding_across": "〉",
        "True_Misconception:Additive": "『",
        "True_Misconception:Base_rate": "』",
        "True_Misconception:Definition": "│",
        "True_Misconception:Denominator-only_change": "─",
        "True_Misconception:Division": "┌",
        "True_Misconception:Duplication": "┐",
        "True_Misconception:Firstterm": "└",
        "True_Misconception:FlipChange": "┘",
        "True_Misconception:Incomplete": "┼",
        "True_Misconception:Incorrect_equivalent_fraction_addition": "█",
        "True_Misconception:Inversion": "▓",
        "True_Misconception:Irrelevant": "▒",
        "True_Misconception:Longer_is_bigger": "£",
        "True_Misconception:Mult": "¥",
        "True_Misconception:Multiplying_by_4": "€",
        "True_Misconception:Not_variable": "₩",
        "True_Misconception:Positive": "©",
        "True_Misconception:Shorter_is_bigger": "®",
        "True_Misconception:Subtraction": "™",
        "True_Misconception:SwapDividend": "♪",
        "True_Misconception:Tacking": "♫",
        "True_Misconception:WNB": "☀",
        "True_Misconception:Whole_numbers_larger": "☁",
        "True_Misconception:Wrong_fraction": "☂",
        "True_Misconception:Wrong_term": "☃",
        "True_Neither:NA": "☎",
    }

    df["completion"] = df["completion"].map(mapping)

    return df

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

def add_all_choice(df: pd.DataFrame) -> pd.DataFrame:
    unique_df = df[["QuestionId", "MC_Answer"]].drop_duplicates()
    question_id_to_all_choice = unique_df.groupby("QuestionId")["MC_Answer"].apply(list)

    df["AllChoice"] = df["QuestionId"].map(question_id_to_all_choice)
    return df

def format_input(row) -> str:
    return PROMPT_FORMAT.format(
        QuestionText=row["QuestionText"],
        AllChoice=row["AllChoice"], # NOTE: listのまま入れている。
        CorrectChoice=row["correct_choice"],
        MC_Answer=row["MC_Answer"],
        Correct="Yes" if row["is_correct"] else "No",
        StudentExplanation=row["StudentExplanation"],
    )

def add_compeltion_token(
        model: AutoModelForCausalLM,
        tokenizer: PreTrainedTokenizer,
        completions: list[str]
    ) -> PreTrainedTokenizer:
    special_tokens_dict = {"additional_special_tokens": completions}
    tokenizer.add_special_tokens(special_tokens_dict)
    print(f"Added {len(completions)} special tokens.")

    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)} tokens.")

    return model, tokenizer

if __name__ == "__main__":
    # Pathの指定
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, help='ディレクトリのパス')
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
    train = add_is_correct_and_correct_choice(train)
    train = add_all_choice(train)
    train["prompt"] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train["prompt"].values[0])

    if DEBUG:
        train = train.sample(100, random_state=SEED).reset_index(drop=True)

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
        attn_implementation="flash_attention_2", # A100なら動くかも
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    all_completions = train["completion"].unique().tolist()
    
    with open(f"{UPLOAD_PATH}/all_completions.json", "w", encoding="utf-8") as f:
        json.dump(all_completions, f, ensure_ascii=False, indent=2)
    print(f"Saved all_completions to {UPLOAD_PATH}/all_completions.json")


    # model, tokenizer = add_compeltion_token(model, tokenizer, all_completions)

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
        # load_best_model_at_end=True,
        # metric_for_best_model="eval_loss",
        packing=True # A100なら動くかも
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # peft_config=lora_config,
    )

    trainer.train()

    # 保存
    trainer.save_model(UPLOAD_PATH)
    tokenizer.save_pretrained(UPLOAD_PATH)