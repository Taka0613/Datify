import streamlit as st
from config import (
    MODEL_PATH,
    DATA_PATH,
    SKIP_ROWS,
    USE_COLUMNS,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    TRAIN_TEST_SPLIT_RATIO,
    XGBOOST_PARAMS,
)
from data_loader import load_data, add_features, split_data
from model_utils import train_xgboost, evaluate_model, save_model_as_py
from feature_importance import (
    compute_shap_values,
    compute_permutation_importance,
    plot_shap_summary,
    plot_feature_importance,
)

# App Title
st.title("XGBoost Model Trainer with Feature Importance")

# Sidebar Inputs
st.sidebar.header("Settings")
data_path = st.sidebar.text_input("Data Path", DATA_PATH)
model_path = st.sidebar.text_input("Model Path", MODEL_PATH)
train_test_split = st.sidebar.slider(
    "Train-Test Split Ratio", 0.1, 0.9, TRAIN_TEST_SPLIT_RATIO, 0.05
)
importance_method = st.sidebar.selectbox(
    "Feature Importance Method", ["SHAP", "Permutation"]
)

# Run the workflow
if st.button("Run Training"):
    # Step 1: Load and preprocess data
    st.text("Loading and preprocessing data...")
    data = load_data(data_path, SKIP_ROWS, USE_COLUMNS)
    data = add_features(data)
    st.write("Sample Data:", data.head())

    # Step 2: Split data into training and testing sets
    X_train, y_train, X_test, y_test = split_data(
        data, FEATURE_COLUMNS, TARGET_COLUMN, train_test_split
    )
    st.write(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

    # Step 3: Train the XGBoost model
    st.text("Training the XGBoost model...")
    model = train_xgboost(X_train, y_train, XGBOOST_PARAMS)

    # Step 4: Evaluate the model
    st.text("Evaluating the model...")
    predictions, rmse = evaluate_model(model, X_test, y_test)
    st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Step 5: Save the model as a Python script
    save_model_as_py(model, model_path, FEATURE_COLUMNS, XGBOOST_PARAMS)
    st.write(f"Model saved to: `{model_path}`")

    # Step 6: Feature Importance Calculation
    st.subheader("Feature Importance")
    if importance_method == "SHAP":
        shap_values, shap_importances = compute_shap_values(model, X_train)
        st.write("SHAP Feature Importances")
        st.write(shap_importances)

        # SHAP Summary Plot
        st.text("SHAP Summary Plot")
        fig = plot_shap_summary(shap_values, X_train)
        st.pyplot(fig)
    else:
        permutation_importances = compute_permutation_importance(
            model, X_train, y_train
        )
        st.write("Permutation Feature Importances")
        st.write(permutation_importances)

        # Permutation Importance Plot
        st.text("Permutation Importance Plot")
        fig = plot_feature_importance(permutation_importances, method="Permutation")
        st.pyplot(fig)

    # Step 7: Display Predictions
    st.subheader("Predictions vs Actuals")
    results = X_test.copy()
    results["Actual"] = y_test
    results["Predicted"] = predictions
    st.write(results)

    # Plot Predictions
    st.line_chart(results[["Actual", "Predicted"]])
