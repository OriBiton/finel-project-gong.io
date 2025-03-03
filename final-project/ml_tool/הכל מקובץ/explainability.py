#!/usr/bin/env python
# coding: utf-8

# In[1]:


import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, VotingClassifier, VotingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMClassifier, LGBMRegressor

def preprocess_data_for_shap(X, trained_features=None):
    """
    מטפל בנתונים לפני שליחתם ל-SHAP, כולל התאמת העמודות לשמות הפיצ'רים המקוריים.
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(X.shape[1])])

    # שמירה על עמודות מספריות בלבד
    X = X.select_dtypes(include=[np.number])

    # טיפול בערכים חסרים
    X = X.fillna(X.mean())

    # התאמת סדר העמודות לפיצ'רים שהמודל אומן עליהם
    if trained_features is not None:
        missing_cols = set(trained_features) - set(X.columns)
        for col in missing_cols:
            X[col] = 0  # הוספת עמודות חסרות עם ערכים ברירת מחדל
        X = X[trained_features]  # מוודא שהסדר זהה

    return X

def explain_model_with_shap(model, X, feature=None):
    try:
        # עיבוד הנתונים
        X = preprocess_data_for_shap(X)

        # יצירת Explainer אוטומטי
        if isinstance(model, (RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, DecisionTreeClassifier, DecisionTreeRegressor)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (XGBClassifier, XGBRegressor)):
            explainer = shap.Explainer(model)
        elif isinstance(model, (LGBMClassifier, LGBMRegressor)):
            explainer = shap.Explainer(model)
        elif isinstance(model, LogisticRegression):
            explainer = shap.LinearExplainer(model, X)
        else:
            raise ValueError("Unsupported model type for SHAP")

        # חישוב ערכי SHAP
        shap_values = explainer(X)

        # אם לא נבחר פיצ'ר, נבחר את הפיצ'ר החשוב ביותר
        if feature is None:
            # חישוב הפיצ'ר החשוב ביותר לפי ממוצע ערכי SHAP מוחלטים
            feature_importance = shap_values.abs.mean(axis=0).values
            most_important_feature = X.columns[np.argmax(feature_importance)]  # הפיצ'ר עם החשיבות הגבוהה ביותר
            feature = most_important_feature
            st.write(f"Selected the most important feature by default: **{feature}**")

        # גרפי SHAP
        st.subheader("גרף סיכום SHAP")
        fig_summary = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, show=False)
        st.pyplot(fig_summary)

        st.subheader("גרף ברים SHAP")
        fig_bar = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X, plot_type="bar", show=False)
        st.pyplot(fig_bar)

        # גרף תלות (Dependence Plot) של הפיצ'ר שנבחר
        st.subheader(f"SHAP Dependence Plot for {feature}")
        fig_dependence, ax = plt.subplots(figsize=(10, 6))  # יצירת אובייקט figure ו-ax
        shap.dependence_plot(feature, shap_values.values, X, ax=ax, show=False)  # כאן מתקנים את הגישה לנתונים
        st.pyplot(fig_dependence)  # קריאה ל-st.pyplot עם figure ו-ax

        

        # הסבר מילולי על הפיצ'רים
        st.write("### Feature Importance Analysis")
        feature_importance = shap_values.abs.mean(axis=0).values
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': feature_importance
        }).sort_values(by='Importance', ascending=False)

        st.write("The table below shows the feature importance based on the mean absolute SHAP values. Features with higher importance have a greater impact on the model's predictions.")
        st.dataframe(importance_df)

        st.write("### Insights on the Features")
        # עכשיו נוודא שההסברים נכונים ומדויקים יותר:
        for feature, importance in zip(importance_df['Feature'], importance_df['Importance']):
            # אם הפיצ'ר חשוב במיוחד, נסביר למה
            if importance > importance_df['Importance'].quantile(0.75):  # בחרנו את הפיצ'רים עם 25% העליונים
                st.write(f"**{feature}:** This feature is highly important with an importance score of {importance:.4f}. It has a significant impact on the model's predictions.")
            else:
                st.write(f"**{feature}:** This feature has a moderate importance score of {importance:.4f}. It contributes to the model's predictions, but not as significantly as others.")

    except Exception as e:
        st.error(f"שגיאה ביצירת ניתוח SHAP: {e}")











#
       















