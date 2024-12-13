import streamlit as st
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
    compute_permutation_importance,
    plot_shap_summary,
    plot_feature_importance,
)

# Initialize session state for search results, selected column, and training flag
if "matching_columns" not in st.session_state:
    st.session_state.matching_columns = []
if "selected_column" not in st.session_state:
    st.session_state.selected_column = None
if "run_training" not in st.session_state:
    st.session_state.run_training = False

# App Title
st.title("App Wonder Palette with Advanced Search and Training")

# Sidebar for Search
st.sidebar.header("Search for Product Data")

# Four search bars for each hierarchy
market_query = st.sidebar.text_input("Search by Market (e.g., 'Market A')")
category_query = st.sidebar.text_input("Search by Category (e.g., 'Category X')")
brand_query = st.sidebar.text_input("Search by Brand (e.g., 'Brand 1')")
item_query = st.sidebar.text_input("Search by Item (e.g., 'Item 101')")

# Search button
if st.sidebar.button("Search"):
    # Perform search and store results in session state
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

    st.session_state.matching_columns = matching_columns  # Save results
    st.session_state.selected_column = None  # Reset selection
    st.session_state.run_training = False  # Reset training flag

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

    # Extract actual column name from selection
    st.session_state.selected_column = selected_option.split(": ", 1)[1]
    st.write(f"You selected: {st.session_state.selected_column}")

    # Run Training button
    if st.button("Run Training"):
        st.session_state.run_training = True  # Flag to start training

# Run training if the flag is set
if st.session_state.run_training and st.session_state.selected_column:
    st.subheader("Model Training and Evaluation")

    # Step 1: Load and preprocess data
    st.text("Loading and preprocessing data...")

    # Dynamically match and load data
    data = load_data(
        DATA_PATH,
        SKIP_ROWS,
        date_column=extract_column_names(DATA_PATH)[0],  # First column as 'date'
        sales_column=st.session_state.selected_column,  # Selected column for 'sales'
    )
    st.write("Sample Data:", data.head())

    # Add features to the data
    st.text("Adding features to the data...")
    data = add_features(data)

    # Step 2: Split data into training and testing sets
    st.text("Splitting data into training and testing sets...")
    X_train, y_train, X_test, y_test = split_data(
        data, FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_TEST_SPLIT_RATIO
    )
    st.write(f"Training Samples: {len(X_train)}, Testing Samples: {len(X_test)}")

    # Step 3: Train the XGBoost model
    st.text("Training the XGBoost model...")
    model = train_xgboost(X_train, y_train, XGBOOST_PARAMS)

    # Step 4: Evaluate the model
    st.text("Evaluating the model...")
    predictions, rmse = evaluate_model(model, X_test, y_test)
    st.success(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Step 5: Save the model as a Python script
    save_model_as_py(model, MODEL_PATH, FEATURE_COLUMNS, XGBOOST_PARAMS)
    st.info(f"Model saved to: `{MODEL_PATH}`")

    # Step 6: Feature Importance Calculation
    st.subheader("Feature Importance")
    importance_method = st.sidebar.selectbox(
        "Feature Importance Method", ["SHAP", "Permutation"]
    )
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
