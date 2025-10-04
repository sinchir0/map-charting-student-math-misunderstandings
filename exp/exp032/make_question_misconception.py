import json

import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/takaito_data")

if __name__ == "__main__":
    # Load the CSV file
    train = pd.read_csv(DATA_PATH / "train.csv")
    
    # QuestionIdに対して、頻度が高い順にMisconceptionを候補としてリスト化(Nanを省いている)
    question_id_to_misconception_candidate = train.groupby("QuestionId")["str_label"].apply(lambda x: x.value_counts().index.tolist())
    
    question_id_to_misconception_candidate_dict = question_id_to_misconception_candidate.to_dict()

    # Save the dictionary as a JSON file
    with open("outputs/question_id_to_misconception_candidate/question_id_to_misconception_candidate_half_label.json", "w") as f:
        json.dump(question_id_to_misconception_candidate_dict, f, indent=4)

