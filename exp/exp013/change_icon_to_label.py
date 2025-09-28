import pandas as pd

val_df = pd.read_csv("outputs/exp013/exp013_20250922065754_validation_results.csv")

inverted_mapping = {
    "■": "False_Correct:NA",
    "□": "False_Misconception:Adding_across",
    "▲": "False_Misconception:Adding_terms",
    "△": "False_Misconception:Additive",
    "▼": "False_Misconception:Base_rate",
    "▽": "False_Misconception:Certainty",
    "◆": "False_Misconception:Definition",
    "◇": "False_Misconception:Denominator-only_change",
    "○": "False_Misconception:Division",
    "●": "False_Misconception:Duplication",
    "★": "False_Misconception:Firstterm",
    "☆": "False_Misconception:FlipChange",
    "♦": "False_Misconception:Ignores_zeroes",
    "♥": "False_Misconception:Incomplete",
    "♠": "False_Misconception:Incorrect_equivalent_fraction_addition",
    "♣": "False_Misconception:Interior",
    "§": "False_Misconception:Inverse_operation",
    "†": "False_Misconception:Inversion",
    "‡": "False_Misconception:Irrelevant",
    "※": "False_Misconception:Longer_is_bigger",
    "∞": "False_Misconception:Mult",
    "±": "False_Misconception:Multiplying_by_4",
    "≠": "False_Misconception:Not_variable",
    "≈": "False_Misconception:Positive",
    "√": "False_Misconception:Scale",
    "∑": "False_Misconception:Shorter_is_bigger",
    "∏": "False_Misconception:Subtraction",
    "∆": "False_Misconception:SwapDividend",
    "Ω": "False_Misconception:Tacking",
    "μ": "False_Misconception:Unknowable",
    "∂": "False_Misconception:WNB",
    "→": "False_Misconception:Whole_numbers_larger",
    "←": "False_Misconception:Wrong_Fraction",
    "↑": "False_Misconception:Wrong_Operation",
    "↓": "False_Misconception:Wrong_fraction",
    "↔": "False_Misconception:Wrong_term",
    "↕": "False_Neither:NA",
    "〈": "True_Correct:NA",
    "〉": "True_Misconception:Adding_across",
    "『": "True_Misconception:Additive",
    "』": "True_Misconception:Base_rate",
    "│": "True_Misconception:Definition",
    "─": "True_Misconception:Denominator-only_change",
    "┌": "True_Misconception:Division",
    "┐": "True_Misconception:Duplication",
    "└": "True_Misconception:Firstterm",
    "┘": "True_Misconception:FlipChange",
    "┼": "True_Misconception:Incomplete",
    "█": "True_Misconception:Incorrect_equivalent_fraction_addition",
    "▓": "True_Misconception:Inversion",
    "▒": "True_Misconception:Irrelevant",
    "£": "True_Misconception:Longer_is_bigger",
    "¥": "True_Misconception:Mult",
    "€": "True_Misconception:Multiplying_by_4",
    "₩": "True_Misconception:Not_variable",
    "©": "True_Misconception:Positive",
    "®": "True_Misconception:Shorter_is_bigger",
    "™": "True_Misconception:Subtraction",
    "♪": "True_Misconception:SwapDividend",
    "♫": "True_Misconception:Tacking",
    "☀": "True_Misconception:WNB",
    "☁": "True_Misconception:Whole_numbers_larger",
    "☂": "True_Misconception:Wrong_fraction",
    "☃": "True_Misconception:Wrong_term",
    "☎": "True_Neither:NA",
}

ground_truth = [
    inverted_mapping[completion] for completion in val_df["completion"].tolist()
]

val_df["completion"] = ground_truth

prediction = [
    [inverted_mapping[pred] for pred in eval(prediction)]
    for prediction in val_df["prediction"].tolist()
]

val_df["prediction"] = prediction

val_df.to_csv("outputs/exp013_20250922065754_results_converted.csv", index=False)