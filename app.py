import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from config import (
    MODEL_PATH,
    DATA_PATH,
    SKIP_ROWS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
    XGBOOST_PARAMS,
)
from data_loader import extract_column_names, load_data, add_features, split_data
from model_utils import train_xgboost, evaluate_model, save_model_as_py
from feature_importance import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
)
import os

# Initialize session state
if "matching_columns" not in st.session_state:
    st.session_state.matching_columns = []
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
if "run_training" not in st.session_state:
    st.session_state.run_training = False
if "model" not in st.session_state:
    st.session_state.model = None
if "shap_values" not in st.session_state:
    st.session_state.shap_values = None
if "X_train" not in st.session_state:
    st.session_state.X_train = None

# App Title
st.title("App Wonder Palette with Explainable AI and Export")

# Sidebar for Search
st.sidebar.header("Search for Product Data")
market_query = st.sidebar.text_input("Search by Market (e.g., 'Market A')")
category_query = st.sidebar.text_input("Search by Category (e.g., 'Category X')")
brand_query = st.sidebar.text_input("Search by Brand (e.g., 'Brand 1')")
item_query = st.sidebar.text_input("Search by Item (e.g., 'Item 101')")

if st.sidebar.button("Search"):
    column_names = extract_column_names(DATA_PATH)
    matching_columns = column_names

    if market_query:
        matching_columns = [
            col for col in matching_columns if market_query.lower() in col.lower()
        ]
    if category_query:
        matching_columns = [
            col for col in matching_columns if category_query.lower() in col.lower()
        ]
    if brand_query:
        matching_columns = [
            col for col in matching_columns if brand_query.lower() in col.lower()
        ]
    if item_query:
        matching_columns = [
            col for col in matching_columns if item_query.lower() in col.lower()
        ]

    st.session_state.matching_columns = matching_columns
    st.session_state.selected_column = None
    st.session_state.run_training = False

# Display search results
if st.session_state.matching_columns:
    st.subheader("Search Results")
    matching_columns_with_index = [
        f"{i}: {col}"
        for i, col in enumerate(st.session_state.matching_columns, start=1)
    ]
    selected_option = st.selectbox(
        "Select a column to train on:", matching_columns_with_index
    )
    st.session_state.selected_column = selected_option.split(": ", 1)[1]
    st.write(f"You selected: {st.session_state.selected_column}")

    if st.button("Run Training"):
        st.session_state.run_training = True

# Run training if flag is set
if st.session_state.run_training and st.session_state.selected_column:
    st.subheader("Model Training and Evaluation")

    date_column = extract_column_names(DATA_PATH)[0]
    sales_column = st.session_state.selected_column
    data = load_data(DATA_PATH, SKIP_ROWS, date_column, sales_column)
    data = add_features(data)

    X_train, y_train, X_test, y_test = split_data(
        data, FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_TEST_SPLIT_RATIO
    )

    model = train_xgboost(X_train, y_train, XGBOOST_PARAMS)
    st.session_state.model = model
    st.session_state.X_train = X_train

    shap_values, _ = compute_shap_values(model, X_train)
    st.session_state.shap_values = shap_values

    predictions, rmse = evaluate_model(model, X_test, y_test)
    st.success(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    save_model_as_py(model, MODEL_PATH, FEATURE_COLUMNS, XGBOOST_PARAMS)
    st.info(f"Model saved to: `{MODEL_PATH}`")

# Explainable AI Section
if st.session_state.model and st.session_state.shap_values is not None:
    st.subheader("Explainable AI")
    mode = st.radio("Select Mode", ["Simple", "Advanced"])

    if mode == "Simple":
        st.write("Top 3 Features Impacting Predictions:")
        shap_summary = plot_shap_summary(
            st.session_state.shap_values, st.session_state.X_train, top_n=3
        )
        st.pyplot(shap_summary)

    elif mode == "Advanced":
        st.write("SHAP Summary Plot:")
        shap_summary = plot_shap_summary(
            st.session_state.shap_values, st.session_state.X_train
        )
        st.pyplot(shap_summary)

        feature = st.selectbox("Select Feature for Dependence Plot", FEATURE_COLUMNS)
        dependence_plot = plot_shap_dependence(
            st.session_state.shap_values, st.session_state.X_train, feature
        )
        st.pyplot(dependence_plot)

    # Export Functionality
    st.subheader("Export Insights")
    export_format = st.selectbox("Select Export Format", ["PDF", "PNG", "CSV"])
    export_button = st.button("Export")

    if export_button:
        output_dir = "exports"
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"shap_summary.{export_format.lower()}")

        if export_format == "PDF":
            shap_summary.savefig(file_path, format="pdf")
        elif export_format == "PNG":
            shap_summary.savefig(file_path, format="png")
        elif export_format == "CSV":
            pd.DataFrame(st.session_state.shap_values).to_csv(file_path, index=False)

        st.success(f"File exported to `{file_path}`")
