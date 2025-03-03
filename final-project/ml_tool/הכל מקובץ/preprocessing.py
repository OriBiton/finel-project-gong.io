#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
from joblib import Parallel, delayed
import streamlit as st

class AutomaticDataCleaner:
    def __init__(self, data: pd.DataFrame):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data
        



        # חפש את העמודה 'count' בצורה לא רגישת לאותיות
        count_col = [col for col in self.data.columns if 'count' in col.lower()]

        # אם העמודה קיימת, בצע את השכפול והמחיקה
        if count_col:
            count_col = count_col[0]  # אם יש יותר מעמודה אחת, קח את הראשונה

            # שכפול השורות לפי הערכים בעמודת count
            self.data = self.data.loc[self.data.index.repeat(self.data[count_col])].reset_index(drop=True)

            # מחיקת העמודה count לאחר השכפול
            self.data = self.data.drop(columns=[count_col])


        # חפש עמודות שמכילות את המילה 'ID' (כולל אם יש רק 'ID', או קו תחתון או רווח לפני 'ID')
        id_columns = [col for col in self.data.columns if pd.Series(col).str.contains(r'(^id$|(_| )id$)', case=False, regex=True).any()]

        # אם יש עמודות כאלה, שנה את הסוג שלהן ל-object
        if id_columns:
            self.data[id_columns] = self.data[id_columns].astype('object')


        




    

    def handle_missing_values(self, strategy="median", categorical_strategy="most_frequent"):
        numeric_cols = self.data.select_dtypes(include=["float64", "int64"])
        if not numeric_cols.empty:
            imputer_num = SimpleImputer(strategy=strategy)
            self.data[numeric_cols.columns] = imputer_num.fit_transform(numeric_cols)

        categorical_cols = self.data.select_dtypes(include=["object", "category"])
        if not categorical_cols.empty:
            imputer_cat = SimpleImputer(strategy=categorical_strategy)
            self.data[categorical_cols.columns] = imputer_cat.fit_transform(categorical_cols)

        print("Missing values handled successfully.")

    def remove_outliers_for_column(self, col, contamination=0.3):
        """
        Function to handle outliers for a single column, to run in parallel.
        """
        column_data = self.data[col]
        if column_data.isnull().sum() > len(column_data) * 0.1:  # Skip columns with too many missing values
            return None

        if column_data.nunique() < 10:
            return None

        try:
            stat, p_value = stats.shapiro(column_data.dropna())
            if p_value > 0.05:
                z_scores = np.abs(stats.zscore(column_data.dropna()))
                outliers_zscore = z_scores > 3
                self.data.loc[outliers_zscore, col] = np.nan
                return f"Column '{col}' handled with Z-score"
            else:
                Q1 = column_data.quantile(0.25)
                Q3 = column_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers_iqr = (column_data < (Q1 - 1.5 * IQR)) | (column_data > (Q3 + 1.5 * IQR))
                self.data.loc[outliers_iqr, col] = np.nan
                iso_forest = IsolationForest(contamination=contamination, random_state=42)
                outliers_iforest = iso_forest.fit_predict(column_data.values.reshape(-1, 1)) == -1
                self.data.loc[outliers_iforest, col] = np.nan
                return f"Column '{col}' handled with IQR and Isolation Forest"
        except ValueError:
            return f"Column '{col}' failed the Shapiro-Wilk test."

    def remove_outliers(self, contamination=0.3):
        """
        Use parallel processing for outlier detection on multiple columns.
        """
        numeric_cols = self.data.select_dtypes(include=[np.number])
        results = Parallel(n_jobs=-1)(delayed(self.remove_outliers_for_column)(col, contamination) 
                                      for col in numeric_cols.columns)
        # Filter out None results
        results = [result for result in results if result is not None]
        return results

    def clean_data(self, handle_missing=True, detect_outliers=True, remove_duplicates=True):
        print("Starting data cleaning...")

        if remove_duplicates:
            self.remove_duplicates()

        if handle_missing:
            self.handle_missing_values()

        if detect_outliers:
            outlier_results = self.remove_outliers()
            for result in outlier_results:
                print(result)

        print("Data cleaning complete!")
        return self.data

def optimize_and_convert(df):
    """
    Convert categorical columns to categories and dummy variables.
    Returns the dataframe after conversion and the remaining column names.
    """
    # Convert object columns to categories
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() == len(df):  # Column with unique values for each row
            df = df.drop(columns=[col])  # Drop columns with unique values
        else:
            df[col] = df[col].astype('category')

    # Convert to dummy variables (one-hot encoding)
    df_dummies = pd.get_dummies(df, drop_first=True)
    
    # Convert boolean columns to integers (True = 1, False = 0)
    for col in df_dummies.select_dtypes(include=[bool]).columns:
        df_dummies[col] = df_dummies[col].astype(int)

    return df_dummies, df_dummies.columns





