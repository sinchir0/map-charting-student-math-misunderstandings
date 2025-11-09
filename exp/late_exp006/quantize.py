from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, Dataset
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048
SEED = 42
USE_FOLD = 0
DATA_PATH = Path("data")
DEBUG = False
FOLD_PATH = Path("outputs/fold/stratified_folds.json")
MODEL_ID = "outputs/late_exp004/full_fine_tuning"
EXP_NAME = "late_exp006"
NOW = datetime.now().strftime("%Y%m%d-%H%M%S")

def make_train() -> Dataset:
    train = pd.read_csv(DATA_PATH / "train.csv")

    if DEBUG:
        train = train.sample(100, random_state=SEED).reset_index(drop=True)

    fold_dict = json.load(open(FOLD_PATH))
    train["fold"] = train["row_id"].astype(str).map(fold_dict)

    train["messages"] = (
        train["QuestionText"].fillna("NA") + " " + train["MC_Answer"].fillna("NA") + " " + train["StudentExplanation"].fillna("NA") + " " + train["Category"].fillna("NA") + " " + train["Misconception"].fillna("NA")
    )
    
    train_df = train[train["fold"] != USE_FOLD].reset_index(drop=True)

    # datasets 形式で読み込み
    train_dataset = Dataset.from_pandas(train_df)
    
    return train_dataset

PROMPT_FORMAT = """\
You are a specialist in identifying the types of misunderstandings that arise from students’ answers to math problems.
Based on the information provided below, please determine what kind of misunderstanding the student has.

Question: {QuestionText}
Answer: {MC_Answer}
Correct: {Correct}
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

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # Load dataset.
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split=f"train_sft[:{NUM_CALIBRATION_SAMPLES}]")
    ds = ds.shuffle(seed=42)

    # ds = make_train()
    train = pd.read_csv(DATA_PATH / "train.csv")
    train = make_completion(train)
    train = change_completion_to_one_token(train)
    train = add_is_correct(train)
    train["prompt"] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train["prompt"].values[0])
    
    # prompt列, completions列を、messages ([{"content": text, "role": "user"}, {"content": text, "role": "assistant"}])形式に変更する
    train["messages"] = train.apply(lambda row: [
        {"content": row["prompt"], "role": "user"},
        {"content": row["completion"], "role": "assistant"},
    ], axis=1)

    ds = Dataset.from_pandas(train)
    ds = ds.shuffle(seed=42)

    ds = ds.select(range(NUM_CALIBRATION_SAMPLES))
    
    # Preprocess the data into the format the model is trained with.
    def preprocess(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False,)}
    # def preprocess(example):
    #     return {"text": example["messages"]}
    ds = ds.map(preprocess)

    # Tokenize the data (be careful with bos tokens - we need add_special_tokens=False since the chat_template already added it).
    # def tokenize(sample):
    #     return tokenizer(sample["text"], padding=False, max_length=MAX_SEQUENCE_LENGTH, truncation=True, add_special_tokens=False)
    # ds = ds.map(tokenize, remove_columns=ds.column_names)

    # Configure the quantization algorithm to run.
    recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

    # Apply quantization.
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=MAX_SEQUENCE_LENGTH,
        num_calibration_samples=NUM_CALIBRATION_SAMPLES,
    )

    # Save to disk compressed.
    SAVE_DIR = "outputs/" + EXP_NAME + "-" + MODEL_ID.rstrip("/").split("/")[-1] + "-" + NOW + "-W4A16-G128"
    model.save_pretrained(SAVE_DIR, save_compressed=True)
    tokenizer.save_pretrained(SAVE_DIR)