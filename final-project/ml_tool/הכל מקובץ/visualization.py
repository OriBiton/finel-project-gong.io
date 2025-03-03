#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def auto_visualize(df, target_col=None):
    """
    פונקציה לניתוח גרפי חכם של דאטה עבור Streamlit
    :param df: DataFrame של פנדס
    :param target_col: שם עמודת המטרה (אם קיימת). ניתן להשאיר כ-None
    """
    sns.set(style="whitegrid", palette="pastel")
    st.write("Target column selected:", target_col)



    # הצגת מדדים תיאוריים
    st.write("== Descriptive Statistics ==")
    st.write(df.describe().transpose())
    


    # ניתוח עמודות מספריות
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    st.write(numeric_cols)
    for col in numeric_cols:
        if df[col].isnull().any():
            st.warning(f"Column {col} contains missing values.")
            df[col] = df[col].fillna(df[col].mean())  # להחליף את הערכים החסרים בממוצע, למשל

        fig, ax = plt.subplots(figsize=(10, 6))
        plt.hist(df[col])  # ללא KDE
        plt.title(f'Histogram of {col}')
        plt.xticks(rotation=90)  # סיבוב התוויות של ה-x ב-90 מעלות
        st.pyplot(fig)


        # אם יש עמודת מטרה, נציג בוקס-פלוט
        if target_col and target_col in df.columns:
            if df[target_col].dtype == 'object':  # אם עמודת המטרה היא לא קטגורית
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=df[target_col], y=df[col], data=df, palette="viridis", ax=ax)
                ax.set_title(f'Boxplot of {col} by {target_col}')
                plt.xticks(rotation=90)  # סיבוב התוויות של ה-x ב-90 מעלות
                st.pyplot(fig)

    # Heatmap של קורלציות עבור עמודות מספריות
    if len(numeric_cols) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = df[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title('Correlation Matrix')
        st.pyplot(fig)

    # ניתוח עמודות קטגוריות
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        if len(df[col].unique())<=20:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.countplot(x=col, data=df)
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=90)  # סיבוב התוויות של ה-x ב-90 מעלות
            st.pyplot(fig)

        # אם יש עמודת מטרה, נציג את הקשר בין העמודה לעמודת המטרה
        if target_col and target_col in df.columns:
            if df[target_col].dtype == 'object':
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.countplot(x=df[col], hue=df[target_col], data=df, palette="viridis", ax=ax)
                ax.set_title(f'{col} Distribution by {target_col}')
                plt.xticks(rotation=90)  # סיבוב התוויות של ה-x ב-90 מעלות
                st.pyplot(fig)
        
        if target_col and target_col in df.columns:
            if df[target_col].dtype != 'object' and len(df[col].unique())<=20:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.boxplot(x=df[col], y=df[target_col], data=df, palette="viridis", ax=ax)
                ax.set_title(f'Boxplot of {target_col} by {col}')
                plt.xticks(rotation=90)  # סיבוב התוויות של ה-x ב-90 מעלות
                st.pyplot(fig)

        

    






    


    


