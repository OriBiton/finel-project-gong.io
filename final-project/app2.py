import streamlit as st
import pandas as pd
from ml_tool.visualization import auto_visualize
from ml_tool.training import build_and_train_model
from ml_tool.explainability import explain_model_with_shap
from run_all import run_all
from ml_tool.clustering import find_patterns_and_importance
from ml_tool.preprocessing import AutomaticDataCleaner
import os

# Function for navigation between pages
def navigate_page(page_name):
    st.session_state.current_page = page_name

# Home page
def home_page():
    st.title("Welcome to My Machine Learning App")
    st.write("Discover critical insights and unlock the full potential of your data!")

    # Add an image (replace 'your_image.png' with your actual image file path)
    st.image(os.path.join(os.getcwd(),'static','Blank board.png'))
    # Navigation button
    if st.button("Get Started"):
        navigate_page("Upload Data")
        st.rerun()

# Upload data page
def upload_data_page():
    st.title("Upload Your Dataset")
    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type="csv")

    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            if data.empty:
                st.error("The uploaded dataset is empty. Please upload a valid file.")
                return

            # Clean the data using AutomaticDataCleaner
            st.write("Cleaning and preparing the data...")
            cleaner = AutomaticDataCleaner(data)
            cleaned_data = cleaner.clean_data()

            # Store cleaned data in session state
            st.session_state.data = cleaned_data

            st.write("Cleaned Dataset preview:")
            st.write(cleaned_data.head())

            # Automatically navigate to the next page
            if st.button("Next page:"):
                navigate_page("Form")
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Form page
def form_page():
    st.title("Dataset Analysis Form")

    

    # Ask if the dataset has a target column
    has_target = st.radio("Does your dataset have a target column?", ("Yes", "No"))
    st.session_state.has_target = has_target

    if has_target == "Yes":
        target_column = st.selectbox("Select the target column", st.session_state.data.columns)
        st.session_state.target_column = target_column

    visualize_data = st.radio("Would you like to see preliminary visualizations?", ("Yes", "No"))
    st.session_state.visualize_data = visualize_data

    # Navigation button
    if st.button("Next"):
        if has_target == "Yes" and visualize_data == "Yes":
            navigate_page("Visualization")
            st.rerun()
        elif has_target == "Yes":
            navigate_page("Model")
            st.rerun()
        else:
            navigate_page("Clustering")
            st.rerun()

# Visualization page
def visualization_page():
    st.title("Data Visualization")

    if "data" not in st.session_state or "target_column" not in st.session_state:
        st.error("No data or target column selected. Please go back to the previous steps.")
        return

    st.write("Visualizing the dataset with target column:", st.session_state.target_column)
    auto_visualize(st.session_state.data, target_col=st.session_state.target_column)

    if st.button("Next"):
        navigate_page("Model")
        st.rerun()

# Model page
def model_page():
    st.title("Model Training and Feature Importance")

    if "data" not in st.session_state or "target_column" not in st.session_state:
        st.error("No data or target column selected. Please go back to the previous steps.")
        return

    try:
        model, processed_X = run_all(st.session_state.data, st.session_state.target_column)
        st.session_state.model = model
        st.session_state.processed_X = processed_X
        st.write("Model trained successfully!")
        st.write("Processed data:")
        st.write(processed_X.head())

        # Explain model with SHAP
        explain_model_with_shap(model, processed_X, processed_X.columns[0])

    except Exception as e:
        st.error(f"An error occurred: {e}")

# Clustering page
def clustering_page():
    st.title("Clustering and Pattern Analysis")

    if "data" not in st.session_state:
        st.error("No data uploaded. Please go back to the previous steps.")
        return

    try:
        find_patterns_and_importance(st.session_state.data)
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Main function
def main():
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"

    # Render the appropriate page based on the current state
    if st.session_state.current_page == "Home":
        home_page()
    elif st.session_state.current_page == "Upload Data":
        upload_data_page()
    elif st.session_state.current_page == "Form":
        form_page()
    elif st.session_state.current_page == "Visualization":
        visualization_page()
    elif st.session_state.current_page == "Model":
        model_page()
    elif st.session_state.current_page == "Clustering":
        clustering_page()

if __name__ == "__main__":
    main()
