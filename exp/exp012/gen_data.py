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
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer
from trl import SFTConfig, SFTTrainer
from datetime import datetime
import json
import wandb

import os
import json

from kaggle.api.kaggle_api_extended import KaggleApi

COMPETITION_NAME = "map-charting-student-math-misunderstandings"
NOW = datetime.now().strftime("%Y%m%d%H%M%S")
EXP_NAME = "exp001_cls"
MODEL_NAME = "Qwen/Qwen3-0.6B"
DATASET_NAME = f"{EXP_NAME}-{MODEL_NAME.split('/')[-1]}-{NOW}"
OUTPUT_PATH = f"outputs/{EXP_NAME}/{NOW}"
CHECKPOINT_PATH = f"{OUTPUT_PATH}/checkpoint"
UPLOAD_PATH = f"{OUTPUT_PATH}/upload"
DATA_PATH = Path("data")
ENV_PATH = Path("env_file")
MAX_LEN = 256
BATCH_SIZE = 8
GRAD_ACCUM = 2
LR = 2e-5
EPOCH = 1
SEED = 42
GEN_PROMPT_FORMAT = """\
You are a specialist in identifying misunderstandings that arise in responses to math problems.

Based on the following “Information for generating StudentExplanation,” think of a possible “StudentExplanation.” Be sure to think carefully along the way about what kind of StudentExplanation would be appropriate. Finally, enclose your generated StudentExplanation in 【】.

# Information for generating StudentExplanation
{request_sample}

Below are an input example and a generation example.

'''
Input Example:
Question: What fraction of the shape is not shaded? Give your answer in its simplest form. [Image: A triangle split into 9 equal smaller triangles. 6 of them are shaded.]
Answer: \( \frac{1}{3} \)
Correct: Yes
Category: True_Misconception:Incomplete

Generation Example:
Thinking process:

I noticed the whole shape is divided into 9 equal smaller triangles.

I counted the shaded ones and found 6.

Then I subtracted to see that 3 were not shaded.

I wrote this as 3/9 and simplified by dividing top and bottom by 3 to get 1/3.

However, to make it “True_Misconception:Incomplete,” the part “3 were not shaded” should not be included in the StudentExplanation.

Finally, it becomes:
【I counted 6 shaded triangles. That makes 3 out of 9 not shaded, which simplifies to 1/3.】
'''

After this, samples where Category and Misconception are the same, and samples where they differ, will be provided. Please use them as references when creating StudentExplanations.

# Sample where Category and Misconception are the same
{same_samples}

# Sample where Category and Misconception are different
{different_samples}
"""
SAMPLE_FORMAT = """\
Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
StudentExplanation: {StudentExplanation}
Category: {Category}
"""
COLS = ["prompt", "completion"]
DEBUG = False
TARGET_COMPLETIONS = [
    "False_Misconception:Adding_terms",
    "False_Misconception:Firstterm",
    "False_Misconception:Multiplying_by_4",
    "True_Misconception:Irrelevant",
    "False_Misconception:FlipChange",
    "False_Misconception:Division",
    "False_Misconception:Definition",
    "False_Misconception:Interior",
    "True_Misconception:Additive",
    "False_Misconception:Longer_is_bigger",
    "False_Misconception:Ignores_zeroes",
    "False_Misconception:Base_rate",
    "False_Misconception:Inverse_operation",
    "False_Misconception:Certainty",
    "True_Misconception:Shorter_is_bigger",
    "True_Misconception:Firstterm",
    "True_Misconception:Wrong_term",
    "True_Misconception:Incomplete",
    "True_Misconception:SwapDividend",
    "True_Misconception:Mult",
    "True_Misconception:WNB",
    "False_Misconception:Incorrect_equivalent_fraction_addition",
    "True_Misconception:Wrong_fraction",
    "False_Misconception:Shorter_is_bigger",
    "False_Misconception:Wrong_Operation",
    "True_Misconception:Duplication",
    "True_Misconception:Division",
    "True_Misconception:Inversion",
    "True_Misconception:Denominator-only_change",
    "True_Misconception:FlipChange",
    "True_Misconception:Definition",
    "True_Misconception:Multiplying_by_4",
    "True_Misconception:Positive",
    "True_Misconception:Subtraction",
    "True_Misconception:Incorrect_equivalent_fraction_addition",
    "True_Misconception:Not_variable",
    "True_Misconception:Base_rate",
    "True_Misconception:Whole_numbers_larger",
    "True_Misconception:Adding_across",
    "True_Misconception:Longer_is_bigger",
]

SAME_SAMPLE_NUM = 3
DIFF_SAMPLE_NUM = 3

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
    return SAMPLE_FORMAT.format(
        QuestionText=row["QuestionText"],
        MC_Answer=row["MC_Answer"],
        Correct="Yes" if row["is_correct"] else "No",
        StudentExplanation=row["StudentExplanation"],
        Category=row["Category"],
    )

def add_completion_token(
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
    load_dotenv(f"{ENV_PATH}/.env")
    seed_everything(SEED)

    os.makedirs(UPLOAD_PATH, exist_ok=True)

    train = pd.read_csv(DATA_PATH / "train.csv")

    train = make_completion(train)
    train = add_is_correct(train)
    
    train["sample_info"] = train.apply(format_input, axis=1)

    # TODO: Question, Answer, Correct の組み合わせと、Category の組み合わせのproductを取るようにし、
    # 質問に対して、全ての誤解が少なくともn個は存在する状態にする。

    for target in TARGET_COMPLETIONS:
        same_df = train[train["completion"] == target]
        diff_df = train[train["completion"] != target]
    
        same_df_sampled = same_df.sample(n=SAME_SAMPLE_NUM)
        diff_df_sampled = diff_df.sample(n=DIFF_SAMPLE_NUM)

        same_sample_info = "\n\n".join([same_df_sampled.sample()["sample_info"].values[0] for _ in range(SAME_SAMPLE_NUM)])
        diff_sample_info = "\n\n".join([diff_df_sampled.sample()["sample_info"].values[0] for _ in range(DIFF_SAMPLE_NUM)])

        request_text = GEN_PROMPT_FORMAT.format(
            request_sample="",
            same_samples=same_sample_info,
            different_samples=diff_sample_info,
        )

        pass