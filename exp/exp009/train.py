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
EXP_NAME = "exp009_improve_prompt"
MODEL_NAME = "Qwen/Qwen3-8B"
FOLD_PATH = Path("outputs/fold/stratified_folds.json")
DATA_PATH = Path("data")
ENV_PATH = Path("env_file")
MAX_LEN = 1536
# BATCH_SIZE = 8
BATCH_SIZE = 6
GRAD_ACCUM = 2
LR = 2e-5
EPOCH = 1
SEED = 42
PROMPT_FORMAT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
Student Explanation: {StudentExplanation}

Next, I will provide the names of the misconception labels along with their explanations. Please use them as a reference.

Adding_across — Adds numerators and denominators straight across (e.g., 1/2 + 1/3 = 2/5) instead of finding a common denominator.

Adding_terms — Improperly combines unlike terms (adds coefficients/variables without respecting like‑term rules).

Additive — Treats a non‑additive relation as additive—for example, adds quantities that should be multiplied or adds terms across a proportion.

Base_rate — Ignores base rates/denominators in ratio or probability problems and focuses only on salient numbers.

Certainty — Asserts absolute certainty (“always/never”) where the context requires conditional or case‑based reasoning.

Definition — Restates a definition or property instead of applying it to compute or reason about the specific problem.

Denominator-only_change — Forms an “equivalent fraction” by changing only the denominator (or only one part), breaking true equivalence.

Division — Chooses division where a different operation is required, or divides in the wrong orientation (swapping dividend/divisor).

Duplication — Double‑counts or repeats a quantity (e.g., multiplies both numerator and denominator by the same factor unnecessarily, or counts the same part twice).

Firstterm — Focuses only on the first term or item, ignoring subsequent terms or conditions in an expression or word problem.

FlipChange — Flips (inverts) and also changes something else simultaneously (e.g., uses “invert and multiply” where it does not apply).

Ignores_zeroes — Drops or cancels zeros improperly (e.g., treats 120 ÷ 30 as “remove zeros” rather than legitimate simplification).

Incomplete — The response is partially correct or relevant but omits a key step, constraint, or justification needed to reach the correct answer.

Incorrect_equivalent_fraction_addition — Creates “equivalent” fractions incorrectly during addition (e.g., scales only one part or scales inconsistently).

Interior — Misapplies a rule about interior quantities (e.g., interior angles/regions) or confuses interior vs. exterior notions.

Inversion — Inverts quantities incorrectly (flips a fraction or reverses a ratio when not warranted).

Inverse_operation — Applies an inverse operation in the wrong context (subtracts when should add, divides when should multiply, etc.).

Irrelevant — The explanation does not address the question asked (off‑topic rationale or unrelated computation).

Longer_is_bigger — Judges size by the “longer written form” (more digits/longer bar) rather than actual numeric magnitude.

Mult — Assumes multiplication is the correct operation or multiplies across mechanically without considering meaning.

Multiplying_by_4 — Applies an unjustified “×4” scaling (e.g., multiplies numerator and denominator by 4 or scales units by 4 without basis).

Not_variable — Treats a variable as a label or fixed symbol rather than as an unknown to operate on algebraically.

Positive — Drops or ignores negative signs; assumes quantities must be positive (e.g., turns ‘−’ into ‘+’ or adds magnitudes).

Scale — Applies scaling inconsistently (e.g., scales one dimension but not another; confuses proportional vs. non‑proportional change).

Shorter_is_bigger — Assumes a shorter written form indicates a larger value (reverse of “Longer_is_bigger”).

Subtraction — Uses subtraction where addition/multiplication is required or subtracts in the wrong order (reverses minuend/subtrahend).

SwapDividend — Swaps dividend/divisor or reverses ratio order, leading to a reciprocal result by mistake.

Tacking — “Tacks on” digits or symbols (e.g., appends zeros or concatenates numbers) instead of performing the intended operation.

Unknowable — Claims the problem cannot be solved with given information, even though it can (premature “insufficient information”).

WNB — Whole Number Bias: treats fractions/ratios like whole numbers (e.g., judges 3/8 > 1/2 because 3>1 or 8>2).

Whole_numbers_larger — Compares by whole‑number cues (larger numerators/denominators ⇒ larger value) instead of actual fraction magnitude.

Wrong_Fraction — Builds an incorrect fraction model (wrong part/whole, swapped numerator/denominator, mismatched units).

Wrong_Operation — Selects the wrong operation for the situation (adds where multiplication is required, etc.).

Wrong_fraction — Same as “Wrong_Fraction”: constructs an incorrect fraction representation (wrong part/whole or swapped roles).

Wrong_term — Selects or manipulates the wrong term in an expression (confuses terms vs. factors, or isolates the wrong part).
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
    train = add_is_correct(train)
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
        # attn_implementation="flash_attention_2", # A100なら動くかも
        device_map="auto",
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    all_completions = train["completion"].unique().tolist()
    
    with open(f"{UPLOAD_PATH}/all_completions.json", "w", encoding="utf-8") as f:
        json.dump(all_completions, f, ensure_ascii=False, indent=2)
    print(f"Saved all_completions to {UPLOAD_PATH}/all_completions.json")


    model, tokenizer = add_compeltion_token(model, tokenizer, all_completions)

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
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss"
        # packing=True # A100なら動くかも
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