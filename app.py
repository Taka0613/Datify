import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from datetime import datetime

# Custom modules
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
    "Root Mean Squared Error (RMSE)": (
        "A metric that measures the average magnitude of prediction errors, penalizing large errors more. "
        "[Learn more](https://statisticsbyjim.com/regression/root-mean-square-error-rmse/)"
    ),
    "Mean Squared Error (MSE)": (
        "The average of squared differences between actual and predicted values. Larger errors contribute more. "
        "[Learn more](https://towardsdatascience.com/understanding-the-mean-squared-error-and-how-it-affects-your-regression-model-4c5d60dbe880)"
    ),
    "Mean Absolute Error (MAE)": (
        "Measures the average absolute differences between actual and predicted values, making it easier to interpret. "
        "[Learn more](https://medium.com/@mireille.tsehaye/mean-absolute-error-mae-explained-d8f18239451f)"
    ),
    "Mean Absolute Percentage Error (MAPE)": (
        "Expresses the prediction error as a percentage of actual values, making it useful for business applications. "
        "[Learn more](https://machinelearningmastery.com/how-to-calculate-mean-absolute-percentage-error-mape-in-python/)"
    ),
    "R² Score (Coefficient of Determination)": (
        "Represents how well the model explains variance in the data, where 1.0 is a perfect fit. "
        "[Learn more](https://www.statology.org/r-squared-in-regression-analysis/)"
    ),
}

# Add glossary section in the sidebar
with st.sidebar.expander("📖 Glossary: Technical Terminology"):
    st.write("Learn more about the key terms used in this app:")
    for term, definition in glossary.items():
        st.markdown(f"**{term}:** {definition}")


# Initialize session state
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
if "run_training" not in st.session_state:
    st.session_state.run_training = False
if "models" not in st.session_state:
    st.session_state.models = {}


# Cache data loading and processing
@st.cache_data
def load_and_process_data(file_path, skip_rows, date_column, sales_column):
    """
    Loads the data, preprocesses it, and adds features.
    """
    data = load_data(file_path, skip_rows, date_column, sales_column)
    data = add_features(data)
    return data


# App Title
st.title("Datify App")

# Tabs for Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🔍 Search",
        "⚙️ Train Model",
        "📊 Explainable AI",
        "📈 Visualization",
        "📊 Model Insights",
    ]
)

import seaborn as sns

# Tab 1: Search and Select Data
with tab1:
    st.header("🔍 Search and Analyze Product Data")

    # Search filters
    market_query = st.text_input("Search by Market")
    category_query = st.text_input("Search by Category")
    brand_query = st.text_input("Search by Brand")
    item_query = st.text_input("Search by Item")

    if st.button("Search"):
        column_names = extract_column_names(DATA_PATH)
        matching_columns = column_names

        # Apply search filters
        for query in [market_query, category_query, brand_query, item_query]:
            if query:
                matching_columns = [
                    col for col in matching_columns if query.lower() in col.lower()
                ]

        st.session_state.matching_columns = matching_columns

    # Display search results
    if "matching_columns" in st.session_state and st.session_state.matching_columns:
        selected_option = st.selectbox(
            "Select a column to analyze:", st.session_state.matching_columns
        )
        if selected_option:
            st.session_state.selected_column = selected_option
            st.success(f"✅ Selected Column: `{st.session_state.selected_column}`")

    # Ensure column is selected before insights
    if st.session_state.selected_column and st.button("Generate Insights"):
        st.session_state.run_insights = True

    # Data Insights Panel
    if "run_insights" in st.session_state and st.session_state.run_insights:
        st.header("📊 Data Insights")

        if not st.session_state.selected_column:
            st.error("⚠️ No column selected.")
        else:
            date_column = extract_column_names(DATA_PATH)[0]
            selected_sales_column = st.session_state.selected_column
            data = load_and_process_data(
                DATA_PATH, SKIP_ROWS, date_column, selected_sales_column
            )

            # 🔹 Display Summary Statistics
            st.subheader("📊 Descriptive Statistics")
            st.dataframe(data.describe())

            # 🔹 Missing Value Analysis
            st.subheader("🚨 Missing Values")
            missing_values = data.isnull().sum()
            missing_percent = (missing_values / len(data)) * 100
            missing_data = pd.DataFrame(
                {"Missing Values": missing_values, "Percentage": missing_percent}
            )
            st.dataframe(missing_data)

            # 🔹 Unique Values Analysis
            st.subheader("🔍 Unique Values per Column")
            unique_counts = data.nunique().to_frame(name="Unique Values")
            st.dataframe(unique_counts)

            # 🔹 Outlier Detection (Boxplot)
            st.subheader("📌 Outlier Detection")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=data[["sales"]], ax=ax)
            ax.set_title("Sales Outliers")
            st.pyplot(fig)

            # 🔹 Sales Trend Visualization
            st.subheader("📈 Sales Trends Over Time")
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(data["date"], data["sales"], label="Sales", color="blue", alpha=0.7)
            ax.plot(
                data["date"],
                data["sales"].rolling(30).mean(),
                label="30-Day Moving Average",
                color="red",
            )
            ax.set_title("Sales Over Time with Moving Average")
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            st.pyplot(fig)

            # 🔹 Sales Distribution Histogram
            st.subheader("📊 Sales Distribution")
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(data["sales"], bins=30, kde=True, ax=ax)
            ax.set_title("Sales Distribution")
            st.pyplot(fig)

            # 🔹 Correlation Heatmap
            st.subheader("📉 Correlation Heatmap")
            correlation_matrix = data.corr()
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(
                correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax
            )
            ax.set_title("Feature Correlations")
            st.pyplot(fig)


# Tab 2: Train Models
with tab2:
    st.header("⚙️ Model Training and Evaluation")

    if "run_insights" not in st.session_state or not st.session_state.run_insights:
        st.warning("⚠️ Please generate data insights first.")
    elif st.session_state.selected_column:
        date_column = extract_column_names(DATA_PATH)[0]
        sales_column = st.session_state.selected_column
        data = load_and_process_data(DATA_PATH, SKIP_ROWS, date_column, sales_column)

        if st.button("Train Models"):
            st.session_state.run_training = True

    if st.session_state.run_training and st.session_state.selected_column:
        with st.spinner("Training models..."):
            X_train, y_train, X_test, y_test = split_data(
                data, FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_TEST_SPLIT_RATIO
            )

            # Train LightGBM
            lgbm_model = LGBMRegressor(
                n_estimators=100, learning_rate=0.1, random_state=42
            )
            lgbm_model.fit(X_train, y_train)
            lgbm_preds = lgbm_model.predict(X_test)
            lgbm_rmse = mean_squared_error(y_test, lgbm_preds, squared=False)

            # Train XGBoost
            xgb_model = XGBRegressor(**XGBOOST_PARAMS)
            xgb_model.fit(X_train, y_train)
            xgb_preds = xgb_model.predict(X_test)
            xgb_rmse = mean_squared_error(y_test, xgb_preds, squared=False)

            # Store models
            st.session_state.models = {
                "LightGBM": {"model": lgbm_model, "rmse": lgbm_rmse},
                "XGBoost": {"model": xgb_model, "rmse": xgb_rmse},
            }
            st.success("Models trained successfully!")

# Tab 3: Explainable AI
with tab3:
    st.header("📊 Explainable AI")

    if "models" in st.session_state and st.session_state.models:
        model_choice = st.radio("Select Model:", ["LightGBM", "XGBoost"])

        if model_choice in st.session_state.models:
            model = st.session_state.models[model_choice]["model"]
            shap_values, _ = compute_shap_values(model, X_train)

            st.subheader("SHAP Summary Plot")
            shap_summary = plot_shap_summary(shap_values, X_train)
            st.pyplot(shap_summary)

            st.subheader("SHAP Dependence Plot")
            feature = st.selectbox("Select Feature", FEATURE_COLUMNS)
            dependence_plot = plot_shap_dependence(shap_values, X_train, feature)
            st.pyplot(dependence_plot)

# Tab 4: Model Insights
with tab5:
    st.header("📊 Model Insights Dashboard")

    if "models" in st.session_state and st.session_state.models:
        model_names = list(st.session_state.models.keys())
        rmses = [st.session_state.models[m]["rmse"] for m in model_names]

        best_model = model_names[rmses.index(min(rmses))]
        st.subheader(f"🏆 Best Model: {best_model} (Lowest RMSE)")

        st.bar_chart({"RMSE": {model: rmse for model, rmse in zip(model_names, rmses)}})

        # Compute SHAP Explainability Score
        shap_importances = {}
        for model_name, model_info in st.session_state.models.items():
            shap_values, _ = compute_shap_values(model_info["model"], X_train)
            shap_importances[model_name] = abs(shap_values.values).mean()

        st.subheader("🔍 Explainability Score")
        st.bar_chart({"SHAP Importance": shap_importances})

        # Trade-Off Analysis
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(rmses, list(shap_importances.values()), color=["blue", "orange"])
        for i, model_name in enumerate(model_names):
            ax.annotate(model_name, (rmses[i], list(shap_importances.values())[i]))

        ax.set_xlabel("RMSE (Lower is Better)")
        ax.set_ylabel("SHAP Explainability Score (Higher is Better)")
        ax.set_title("Trade-Off Between Accuracy and Explainability")
        st.pyplot(fig)

    else:
        st.warning("⚠️ No trained models found.")
