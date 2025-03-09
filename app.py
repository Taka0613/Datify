import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from config import (
    MODEL_PATH,
    DATA_PATH,
    SKIP_ROWS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
    XGBOOST_PARAMS,
)
from data_loader import *
from model_utils import train_xgboost, evaluate_model, save_model_as_py
from feature_importance import (
    compute_shap_values,
    plot_shap_summary,
    plot_shap_dependence,
)
import os
from datetime import datetime

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

# Define glossary terms with links
glossary = {
    "Root Mean Squared Error (RMSE)": (
        "A metric that measures the average magnitude of prediction errors. "
        "[Learn more](https://statisticsbyjim.com/regression/root-mean-square-error-rmse/)"
    ),
    "SHAP Values": (
        "Shapley Additive Explanations are used to interpret machine learning models "
        "by attributing the effect of each feature on predictions. "
        "[Learn more](https://towardsdatascience.com/using-shap-values-to-explain-how-your-machine-learning-model-works-732b3f40e137)"
    ),
    "XGBoost": (
        "An optimized gradient boosting library designed to be highly efficient and flexible. "
        "[Learn more](https://medium.com/@prathameshsonawane/xgboost-how-does-this-work-e1cae7c5b6cb/)"
    ),
    "Feature Importance": (
        "A technique to rank features based on their contribution to the model's predictions. "
        "[Learn more](https://builtin.com/data-science/feature-importance#:~:text=What%20is%20feature%20importance%20simply,a%20machine%20learning%20model's%20predictions./)"
    ),
    "Train-Test Split": (
        "Dividing data into training and testing sets to evaluate model performance. "
        "[Learn more](https://www.shiksha.com/online-courses/articles/train-test-split/#:~:text=A%20train%20test%20split%20is,on%20the%20unseen%20testing%20set./)"
    ),
}

# Add glossary section in the sidebar
with st.sidebar.expander("📖 Glossary: Technical Terminology"):
    st.write("Learn more about the key terms used in this app:")
    for term, definition in glossary.items():
        st.markdown(f"**{term}:** {definition}")


# Cache data loading and processing
@st.cache_data
def load_and_process_data(file_path, skip_rows, date_column, sales_column):
    data = load_data(file_path, skip_rows, date_column, sales_column)
    data = add_features(data)
    return data


# App Title
st.title("Datify app")

# Tabs for Navigation
tab1, tab2, tab3, tab4 = st.tabs(
    ["🔍 Search", "⚙️ Train Model", "📊 Explainable AI", "📈 Visualization"]
)

with tab1:
    st.header("🔍 Search for Product Data")

    # Search filters
    market_query = st.text_input("Search by Market (e.g., 'Market A')")
    category_query = st.text_input("Search by Category (e.g., 'Category X')")
    brand_query = st.text_input("Search by Brand (e.g., 'Brand 1')")
    item_query = st.text_input("Search by Item (e.g., 'Item 101')")

    if st.button("Search"):
        column_names = extract_column_names(DATA_PATH)
        matching_columns = column_names

        # Apply filters dynamically
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
        st.session_state.selected_column = None  # Reset selection
        st.session_state.run_training = False  # Reset training state

    # Display search results
    if st.session_state.matching_columns:
        st.subheader("Search Results")
        selected_option = st.selectbox(
            "Select a column to analyze:", st.session_state.matching_columns
        )

        if selected_option:
            st.session_state.selected_column = selected_option  # Store selection
            st.write(f"✅ Selected Column: `{st.session_state.selected_column}`")

    # Ensure a column is selected before proceeding to insights
    if not st.session_state.selected_column:
        st.warning("⚠️ Please select a column before generating insights.")
    else:
        if st.button("Generate Insights"):
            st.session_state.run_insights = True  # Flag to trigger insights

    # Data Insights Panel (Runs when "Generate Insights" is clicked)
    if "run_insights" in st.session_state and st.session_state.run_insights:
        st.header("📊 Data Insights")

        # Ensure a valid column is selected
        if not st.session_state.selected_column:
            st.error(
                "⚠️ No column selected. Please select a column before running insights."
            )
        else:
            date_column = extract_column_names(DATA_PATH)[0]
            selected_sales_column = st.session_state.selected_column

            # Load data with selected column
            data = load_and_process_data(
                DATA_PATH, SKIP_ROWS, date_column, selected_sales_column
            )

            insights = generate_data_insights(data)

            # Display missing values
            if insights["missing_values"]:
                st.subheader("🔴 Missing Values Detected")
                for col, percentage in insights["missing_values"].items():
                    st.write(f"**{col}**: {percentage:.2f}% missing")
            else:
                st.write("✅ No missing values detected!")

            # Show feature statistics
            st.subheader("📊 Feature Statistics")
            st.dataframe(
                pd.DataFrame.from_dict(insights["feature_statistics"], orient="index")
            )

            # Show correlation with sales
            if "correlation" in insights:
                st.subheader("📈 Correlation with Sales")
                st.bar_chart(pd.Series(insights["correlation"]))

            # Show data distribution plots
            st.subheader("📌 Data Distributions")
            st.pyplot(plot_data_distribution(data))


# Tab 2: Train Model
with tab2:
    st.header("⚙️ Model Training and Evaluation")

    # Ensure that insights were generated first
    if "run_insights" not in st.session_state or not st.session_state.run_insights:
        st.warning("⚠️ Please generate data insights before training the model.")
    elif st.session_state.selected_column:
        date_column = extract_column_names(DATA_PATH)[0]
        sales_column = st.session_state.selected_column
        data = load_and_process_data(DATA_PATH, SKIP_ROWS, date_column, sales_column)

        # Model selection dropdown
        model_choice = st.radio(
            "Select Model:", ["LightGBM (Fast)", "XGBoost (Slower, More Accurate)"]
        )

        if st.button("Train Model"):
            st.session_state.run_training = True  # Flag for training

    # Run Training if initiated
    if st.session_state.run_training and st.session_state.selected_column:
        with st.spinner("Training model..."):
            # Split the data
            X_train, y_train, X_test, y_test = split_data(
                data, FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_TEST_SPLIT_RATIO
            )

            # Select and train model
            if model_choice == "LightGBM (Fast)":
                from lightgbm import LGBMRegressor

                model = LGBMRegressor(
                    n_estimators=100, learning_rate=0.1, random_state=42
                )
            else:
                from xgboost import XGBRegressor

                model = XGBRegressor(**XGBOOST_PARAMS)

            model.fit(X_train, y_train)
            st.session_state.model = model
            st.session_state.X_train = X_train

            # Compute SHAP values
            shap_values, _ = compute_shap_values(model, X_train)
            st.session_state.shap_values = shap_values

            # Make predictions
            predictions = model.predict(X_test)
            rmse = mean_squared_error(y_test, predictions, squared=False)
            st.metric(label="RMSE", value=f"{rmse:.2f}")

            # Save model
            save_model_as_py(model, MODEL_PATH, FEATURE_COLUMNS, XGBOOST_PARAMS)
            st.success(f"Model saved!")

# Tab 3: Explainable AI
with tab3:
    st.header("📊 Explainable AI")

    if st.session_state.model and st.session_state.shap_values is not None:
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

            feature = st.selectbox(
                "Select Feature for Dependence Plot", FEATURE_COLUMNS
            )
            dependence_plot = plot_shap_dependence(
                st.session_state.shap_values, st.session_state.X_train, feature
            )
            st.pyplot(dependence_plot)

# Tab 4: Visualization
with tab4:
    st.header("📈 Data Visualization")

    if st.session_state.selected_column:
        date_column = extract_column_names(DATA_PATH)[0]
        data = load_and_process_data(
            DATA_PATH, SKIP_ROWS, date_column, st.session_state.selected_column
        )

        # Convert date to datetime for Streamlit slider compatibility
        data["date"] = pd.to_datetime(data["date"])
        min_date, max_date = data["date"].min(), data["date"].max()

        date_range = st.slider(
            "Select Date Range",
            min_value=min_date.to_pydatetime(),
            max_value=max_date.to_pydatetime(),
            value=(min_date.to_pydatetime(), max_date.to_pydatetime()),
        )

        filtered_data = data[
            (data["date"] >= date_range[0]) & (data["date"] <= date_range[1])
        ]

        st.subheader("Original Data Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            filtered_data["date"],
            filtered_data["sales"],
            label="Original Data",
            color="blue",
        )
        ax.set_title("Original Sales Data")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)

    if st.session_state.model:
        st.subheader("Prediction Visualization")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(
            data.loc[X_test.index, "date"],
            y_test,
            label="Actual Sales (Test)",
            color="blue",
        )
        ax.plot(
            data.loc[X_test.index, "date"],
            predictions,
            label="Predicted Sales",
            color="orange",
        )
        ax.set_title("Model Predictions vs Actual Sales")
        ax.set_xlabel("Date")
        ax.set_ylabel("Sales")
        ax.legend()
        ax.tick_params(axis="x", rotation=45)
        st.pyplot(fig)
