from auto_round import AutoRound
from datetime import datetime
from transformers import AutoTokenizer
import pandas as pd
from pathlib import Path

DATA_PATH = Path("data")
NOW = datetime.now().strftime("%Y%m%d%H%M%S")
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

    # Load a model (supports FP8/BF16/FP16/FP32)
    model_name_or_path = "outputs/late_exp004/20251108134354/upload"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    train = pd.read_csv(DATA_PATH / "train.csv")
    train = make_completion(train)
    train = change_completion_to_one_token(train)
    train = add_is_correct(train)
    train["prompt"] = train.apply(format_input, axis=1)
    print("Example prompt for our LLM:")
    print(train["prompt"].values[0])

    # texts = [
    #     "Hello, how are you?",
    #     "The capital of France is Paris.",
    #     "Transformers are powerful models."
    # ]

    texts = train["prompt"].tolist()
    # ランダムに512件を取得する
    import random
    random.seed(42)
    texts = random.sample(texts, min(512, len(texts)))

    def make_calib_data(texts, tokenizer):
        input_ids = []
        for t in texts:
            ids = tokenizer(t, return_tensors="pt").input_ids
            input_ids.append(ids)
        return input_ids

    calib_data = make_calib_data(texts, tokenizer)

    # Available schemes: "W2A16", "W3A16", "W4A16", "W8A16", "NVFP4", "MXFP4" (no real kernels), "GGUF:Q4_K_M", etc.
    ar = AutoRound(
        model_name_or_path,
        calib_data=calib_data,
        scheme="W4A16"
    )

    # Highest accuracy (4–5× slower).
    # `low_gpu_mem_usage=True` saves ~20GB VRAM but runs ~30% slower.
    # ar = AutoRound(model_name_or_path, nsamples=512, iters=1000, low_gpu_mem_usage=True)

    # Faster quantization (2–3× speedup) with slight accuracy drop at W4G128.
    # ar = AutoRound(model_name_or_path, nsamples=128, iters=50, lr=5e-3)

    # Supported formats: "auto_round" (default), "auto_gptq", "auto_awq", "llm_compressor", "gguf:q4_k_m", etc.
    ar.quantize_and_save(
        output_dir=f"outputs/late_exp008/{NOW}",
        format="auto_gptq"
    )