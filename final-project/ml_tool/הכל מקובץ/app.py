#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import pandas as pd
from ml_tool.visualization import auto_visualize
from ml_tool.training import build_and_train_model
from ml_tool.explainability import explain_model_with_shap
from run_all import run_all
from ml_tool.clustering import find_patterns_and_importance
from ml_tool.preprocessing import AutomaticDataCleaner  # הוספת פונקציה לניקוי וסידור הנתונים


def main():
    st.title("Machine Learning Tool for Data Analysis and Model Building")

    # File upload
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if data.empty:
                st.error("The uploaded dataset is empty. Please upload a valid file.")
                return

            st.write("Dataset preview:")
            st.write(data.head())

            # Data cleaning
            if 'data_cleaned' not in st.session_state:
                st.write("Cleaning and preparing the data...")
                cleaner = AutomaticDataCleaner(data)
                st.session_state.data_cleaned = cleaner.clean_data()
                st.write("Cleaned Dataset preview:")
                st.write(st.session_state.data_cleaned.head())
            else:
                st.write("Cleaned Dataset preview:")
                st.write(st.session_state.data_cleaned.head())

            # Target column
            has_target = st.radio("Does your dataset have a target column?", ("Select an option", "Yes", "No"))

            if has_target == "Yes":
                target_column = st.selectbox("Select the target column", st.session_state.data_cleaned.columns)
                if target_column:
                    visualize_data = st.radio("Would you like to see visualizations of your data?", ("Select an option", "Yes", "No"))
                    if visualize_data == "Yes":
                        st.write("Target column selected:", target_column)

                        auto_visualize(st.session_state.data_cleaned, target_col=target_column)

                    if st.button("Build Model and Analyze Feature Importance"):
                        try:
                            model, processed_X = run_all(st.session_state.data_cleaned, target_column)
                            st.session_state.model = model
                            st.session_state.X = processed_X
                            st.session_state.feature = processed_X.columns[0]
                            st.write("Processed data for SHAP:")
                            st.write(processed_X.head())

                            explain_model_with_shap(model, processed_X, st.session_state.feature)
                        except Exception as e:
                            st.error(f"An error occurred: {e}")

            elif has_target == "No":
                if st.radio("Would you like to see general insights about your data?", ("Select an option", "Yes", "No")) == "Yes":
                    auto_visualize(st.session_state.data_cleaned)
                if st.button("Analyze Clustering and Feature Importance"):
                    try:
                        find_patterns_and_importance(st.session_state.data_cleaned)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")


if __name__ == '__main__':
    main()










