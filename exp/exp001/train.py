import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from pathlib import Path

# ref: https://www.kaggle.com/code/cdeotte/gemma2-9b-it-cv-0-945

DATA_PATH = Path("data")

train = pd.read_csv(DATA_PATH / "train.csv")

le = LabelEncoder()

train["Misconception"] = train["Misconception"].fillna('NA')
train['target'] = train["Category"] + ":" + train["Misconception"]
train['label'] = le.fit_transform(train['target'])
target_classes = le.classes_
n_classes = len(target_classes)
print(f"Train shape: {train.shape} with {n_classes} target classes")

# Powerful Feature Engineer
# 前提として、ラベル付けが誤っていることがある。
# よって、QuestionIdに対して、MC_AnswerがTrueになっている回答が最も多いものを、真の正解として扱う。
idx = train.apply(lambda row: row.Category.split('_')[0],axis=1)=='True'
correct = train.loc[idx].copy()
correct['c'] = correct.groupby(['QuestionId','MC_Answer']).MC_Answer.transform('count')
correct = correct.sort_values('c',ascending=False)
correct = correct.drop_duplicates(['QuestionId'])
correct = correct[['QuestionId','MC_Answer']]
correct['is_correct'] = 1

train = train.merge(correct, on=['QuestionId','MC_Answer'], how='left')
train.is_correct = train.is_correct.fillna(0)

pass