#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingClassifier, VotingRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor
import numpy as np
import pandas as pd

def detect_problem_type(target_column):
    """
    מזהה את סוג הבעיה:
    - סיווג: אם יש מספר מוגבל של ערכים ייחודיים, והם לא רציפים (לא מדידה על סקאלה)
    - רגרסיה: אם יש יותר מ-10 ערכים ייחודיים, או אם מדובר בנתונים רציפים.
    - מדד סדרתי: אם הערכים בעלי סדר (ordinal) אך מוגבלים.
    - סיווג בינארי: אם הערכים הם 0 ו-1 בלבד.
    """
    
    unique_values = target_column.unique()
    num_unique = len(unique_values)
    
    # אם הערכים הם 0 ו-1 בלבד, נחשיב את זה כבעיה של סיווג בינארי
    if set(unique_values) == {0, 1}:
        return "classification"
    
    # בדוק אם הערכים הם רציפים (האם יש מיתאם מספרי בין הערכים?)
    if np.issubdtype(target_column.dtype, np.number) :
        return "regression"  # אם מדובר במספרים רציפים, אז רגרסיה
     
    # אם יש ערכים קטגוריים רבים, אז סיווג
    else:
        return "classification"


import streamlit as st
import shap
import pandas as pd
from sklearn.metrics import accuracy_score, mean_squared_error

def build_and_train_model(X, y, problem_type):
    """
    בונה מודל, מאמן אותו, מציג מידע בסטרים ומחזיר את המודל הטוב ביותר.
    """
    st.write("🔍 Detecting Problem Type...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    best_model = None
    best_score = None
    best_model_name = ""
    st.write(f"🚀 Problem type detected: **{problem_type}**")
    
    if problem_type == "regression":
        models = {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(),
            "RandomForestRegressor": RandomForestRegressor(),
            "XGBRegressor": XGBRegressor(),
            "GradientBoostingRegressor": GradientBoostingRegressor(),
            "LGBMRegressor": LGBMRegressor(),
            "KNeighborsRegressor": KNeighborsRegressor()
        }

        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Model: {model_name}, MSE: {mse:.4f}")
            
            if best_score is None or mse < best_score:
                best_model = model
                best_score = mse
                best_model_name = model_name

        # Cross Validation with GridSearch for models that support n_estimators
        if isinstance(best_model, (RandomForestRegressor, XGBRegressor, GradientBoostingRegressor, LGBMRegressor)):
            grid_search = GridSearchCV(best_model, param_grid={"n_estimators": [50, 100, 150]}, cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Evaluate with GridSearchCV results
            predictions = best_model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            st.write(f"Best Model from GridSearch: {best_model_name}, MSE: {mse:.4f}")
    
    elif problem_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(solver='lbfgs'),
            "LogisticRegressionCV": LogisticRegressionCV(cv=5),
            "DecisionTreeClassifier": DecisionTreeClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "XGBClassifier": XGBClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "LGBMClassifier": LGBMClassifier(),
            "KNeighborsClassifier": KNeighborsClassifier()
        }

        best_model = None
        best_score = None
        best_model_name = ""
        
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Model: {model_name}, Accuracy: {accuracy:.4f}")
            
            if best_score is None or accuracy > best_score:
                best_model = model
                best_score = accuracy
                best_model_name = model_name

        # Cross Validation with GridSearch for models that support n_estimators
        if isinstance(best_model, (RandomForestClassifier, XGBClassifier, GradientBoostingClassifier, LGBMClassifier)):
            grid_search = GridSearchCV(best_model, param_grid={"n_estimators": [50, 100, 150]}, cv=5)
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_

            # Evaluate with GridSearchCV results
            predictions = best_model.predict(X_test)
            accuracy = accuracy_score(y_test, predictions)
            st.write(f"Best Model from GridSearch: {best_model_name}, Accuracy: {accuracy:.4f}")

            
    
    # הצגת המודל הטוב ביותר והדיוק/MSE שלו
    st.write(f"🎯 **Best Model:** {best_model_name}")
    if problem_type == "regression":
        st.write(f"🔹 **Best Model MSE:** {best_score:.4f}")
    else:
        st.write(f"🔹 **Best Model Accuracy:** {best_score:.4f}")
    

    return best_model, X.columns









