#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import numpy as np
import pandas as pd
from ml_tool.preprocessing import AutomaticDataCleaner, optimize_and_convert
from ml_tool.training import build_and_train_model, detect_problem_type
from ml_tool.clustering import find_patterns_and_importance  # יבוא הפונקציה של הקלאסטרינג
import streamlit as st

def run_all(data, target_column=None, n_clusters=3):
    data_encoded, feature_columns = optimize_and_convert(data)
    
    if target_column:  # אם יש עמודת מטרה, נבצע סיווג
        # סיווג (Supervised Learning)
        X = data_encoded.drop(columns=[target_column])
        y = data_encoded[target_column]
        
        # שמירה על שמות הפיצ'רים
        X = pd.DataFrame(X, columns=X.columns)
        problem_type = detect_problem_type(data[target_column])
        model, used_columns = build_and_train_model(X, y, problem_type)
        
        # החזר את הנתונים עם שמות הפיצ'רים
        X_for_shap = X[used_columns]
        return model, X_for_shap
    else:
        # קלאסטרינג (Unsupervised Learning)
        # נבצע קלאסטרינג ונתח את הפיצ'ר אימפורטנס
        find_patterns_and_importance(data, n_clusters)
        return None, None  # לא מחזירים מודל כשיש קלאסטרינג




