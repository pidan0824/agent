"""Unmodified function bodies extracted from UserMirrorer (MIT); see PROVENANCE.json."""
import numpy as np
import pandas as pd
from collections import OrderedDict

def get_entropy(prob: np.array):
    entropy = -np.sum(prob * np.log2(prob))
    return entropy

def get_prob(choice: dict, choice_cnt: int):
    default_prob = {chr(i + 65): 1e-10 for i in range(choice_cnt)}
    default_prob.update(choice)
    choice = OrderedDict(sorted(default_prob.items(), key=lambda x: x[0])).values()

    prob = np.array(list(choice))
    prob = np.maximum(prob, 1e-10)
    prob = prob / prob.sum()
    return prob

def get_logloss(prob: np.array, item_pos: int):
    label = np.zeros_like(prob)
    if isinstance(item_pos, int):
        label[item_pos] = 1.
    else:
        label[ord(item_pos) - 65] = 1.
    logloss = -np.sum(label * np.log2(prob))
    return logloss

def calculate_metrics(teacher: pd.DataFrame, student: pd.DataFrame) -> pd.DataFrame:
    df = teacher.groupby(level=[0,1]).first()

    df.loc[:, 'teacher_hit'] = teacher['preference'].groupby(level=0).mean()
    df.loc[:, 'student_hit'] = student['preference'].groupby(level=0).mean()
    df.loc[:, "diff_hit"] = df['student_hit'] - df['teacher_hit']

    df.loc[:, 'teacher_entropy'] = teacher['entropy'].groupby(level=[0,1]).mean()
    df.loc[:, 'student_entropy'] = student['entropy'].groupby(level=[0,1]).mean()
    df.loc[:, 'teacher_uncertainty'] = teacher['uncertainty'].groupby(level=[0,1]).mean()
    df.loc[:, 'student_uncertainty'] = student['uncertainty'].groupby(level=[0,1]).mean()
    df.loc[:, 'teacher_data_uncertainty'] = df.apply(lambda x: x["teacher_uncertainty"] - x["teacher_entropy"], axis=1)
    df.loc[:, 'student_data_uncertainty'] = df.apply(lambda x: x["student_uncertainty"] - x["student_entropy"], axis=1)
    
    df.loc[:, 'diff_all_uncertainty'] = df['student_uncertainty'] - df['teacher_uncertainty']
    df.loc[:, 'diff_model_uncertainty'] = df['student_entropy'] - df['teacher_entropy']
    df.loc[:, 'diff_data_uncertainty'] = (df['student_uncertainty'] - df['teacher_uncertainty']) - (df['student_entropy'] - df['teacher_entropy'])

    df.loc[:, 'teacher_logloss'] = teacher['logloss'].groupby(level=[0,1]).mean()
    df.loc[:, 'student_logloss'] = student['logloss'].groupby(level=[0,1]).mean()
    return df
