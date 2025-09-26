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




COMPETITION_NAME = "map-charting-student-math-misunderstandings"
NOW = datetime.now().strftime("%Y%m%d%H%M%S")
EXP_NAME = "exp017_use_map_3"
MODEL_NAME = "Qwen/Qwen3-8B"
FOLD_PATH = Path("outputs/fold/stratified_folds.json")
DATA_PATH = Path("data")
ENV_PATH = Path("env_file")
COMPLETION_PATH = Path("all_completions")
# MAX_LEN = 256
# MAX_LEN = 1024
MAX_LEN = 1152
# BATCH_SIZE = 8
TRAIN_BATCH_SIZE = 6
EVAL_BATCH_SIZE = 1
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
    
    # all_completions = train["completion"].unique().tolist()
    # with open(f"{UPLOAD_PATH}/all_completions.json", "w", encoding="utf-8") as f:
    #     json.dump(all_completions, f, ensure_ascii=False, indent=2)
    # print(f"Saved all_completions to {UPLOAD_PATH}/all_completions.json")

    with open(COMPLETION_PATH / "all_completions.json", "r", encoding="utf-8") as f:
        all_completions = json.load(f)

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

    # completion -> token_id（1トークンであることを前提：special token）
    completion2id: dict[str, int] = {
        c: tokenizer(c)["input_ids"][0] for c in all_completions
    }
    candidate_token_ids: torch.Tensor = torch.tensor(list(completion2id.values()))
    id2completion: dict[int, str] = {v: k for k, v in completion2id.items()}

    def map3_score(pred_top3s: list[list[int]], golds: list[str]) -> float:
        """pred_top3: 各サンプルの上位3 token_id, gold: 正解 completion 文字列"""
        scores = []
        for pred_top3, gold in zip(pred_top3s, golds):
            try:
                rank = pred_top3.index(completion2id[gold]) + 1  # 1-based
                scores.append(1.0 / rank)
            except ValueError: # indexに入らない場合
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    # === 追加: compute_metrics 本体 ===
    # SFTTrainer からは EvalPrediction(predictions, label_ids) が来る。
    # predictions: (bsz, seq_len, vocab), label_ids: (bsz, seq_len) with -100 on prompt
    # def compute_metrics_fn(eval_pred) -> dict[str, float]:
    #     preds_np, labels_np = eval_pred
    #     # numpy -> torch
    #     logits = torch.from_numpy(preds_np)           # (B, T, V)
    #     labels = torch.from_numpy(labels_np)          # (B, T)

    #     B, T, V = logits.shape

    #     # 各サンプルについて「最初のラベル位置」= completion の先頭トークンの位置を特定
    #     # labels != -100 の最初の index
    #     first_label_idx = []
    #     for i in range(B):
    #         idxs = (labels[i] != -100).nonzero(as_tuple=False).squeeze(-1)
    #         if idxs.numel() == 0:
    #             first_label_idx.append(None)
    #         else:
    #             first_label_idx.append(int(idxs[0].item()))

    #     # 候補トークン以外を無視して Top-3 を取得
    #     top3_ids_per_sample: list[list[int]] = []
    #     with torch.no_grad():
    #         for i in range(B):
    #             j = first_label_idx[i]
    #             if j is None:
    #                 top3_ids_per_sample.append([])
    #                 continue

    #             # 位置 j のロジットから候補トークンのみ抽出
    #             # logits[i, j, :] -> (V,)
    #             logit = logits[i, j, :]  # (V,)
    #             # gather で候補だけ取り出す
    #             cand_logits = logit[candidate_token_ids]  # (C,)
    #             # Top-3（C が 3 未満ならその分だけ）
    #             k = min(3, cand_logits.shape[0])
    #             topk_vals, topk_idx = torch.topk(cand_logits, k=k, dim=-1)
    #             # 元の vocab の token_id に戻す
    #             top3_token_ids = candidate_token_ids[topk_idx].tolist()
    #             top3_ids_per_sample.append(top3_token_ids)

    #     # gold completion（順序は val_ds の順 = val_df の順）
    #     gold = val_df["completion"].tolist()[:B]

    #     map3 = map3_score(top3_ids_per_sample, gold)

    #     # ついでに@1（= accuracy）や@2も見たい場合はここで計算して返せる
    #     # acc@1:
    #     # acc1 = map3_score([ids[:1] for ids in top3_ids_per_sample], gold)
    #     # acc2 = map3_score([ids[:2] for ids in top3_ids_per_sample], gold)

    #     return {
    #         "map3": map3,
    #         # "acc@1": acc1,
    #         # "acc@2": acc2,
    #     }

    def compute_metrics_fn(eval_pred) -> dict[str, float]:
        preds_np, labels_np = eval_pred  # both are numpy arrays
        # preds_np: (B, T, V), labels_np: (B, T)

        B, T, V = preds_np.shape

        # labels != -100 の最初の index を求める (numpy で)
        first_label_idx = []
        for i in range(B):
            idxs = np.where(labels_np[i] != -100)[0]
            if idxs.size == 0:
                first_label_idx.append(None)
            else:
                first_label_idx.append(int(idxs[0]))

        # candidate token ids を numpy に
        candidate_token_ids_np = np.array(candidate_token_ids.tolist(), dtype=np.int64)  # C,
        C = candidate_token_ids_np.shape[0]

        top3_ids_per_sample = []

        # for speed: we can vectorize if many j are the same, but simplest is loop per sample
        for i in range(B):
            j = first_label_idx[i]
            if j is None:
                top3_ids_per_sample.append([])
                continue

            # 位置 j のロジットから候補のみ抽出（numpy）
            # preds_np[i, j, :] -> (V,)
            cand_logits = preds_np[i, j, candidate_token_ids_np]  # (C,)

            # Top-k; k = min(3, C)
            k = min(3, C)
            if C <= 1000:
                # small C: use argsort for clarity
                topk_idx_in_cand = np.argsort(-cand_logits)[:k]  # indices in 0..C-1
            else:
                # larger C: argpartition is usually faster
                part = np.argpartition(-cand_logits, k-1)[:k]
                topk_idx_in_cand = part[np.argsort(-cand_logits[part])]

            # map back to original vocab token ids
            topk_token_ids = candidate_token_ids_np[topk_idx_in_cand].tolist()
            top3_ids_per_sample.append(topk_token_ids)

        # gold completion（val_df の順）. 注意: eval_pred の順が val_ds の順になっている前提
        gold = val_df["completion"].tolist()[:B]

        map3 = map3_score(top3_ids_per_sample, gold)

        return {"map3": map3}


    sft_config = SFTConfig(
        output_dir=CHECKPOINT_PATH,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
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
        metric_for_best_model="map3",
        packing=True # A100なら動くかも
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # peft_config=lora_config,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    # 保存
    trainer.save_model(UPLOAD_PATH)
    tokenizer.save_pretrained(UPLOAD_PATH)