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

if __name__ == "__main__":
    # Step 1: Load and preprocess data
    data = load_data(DATA_PATH, SKIP_ROWS, USE_COLUMNS)
    data = add_features(data)

    # Step 2: Split data into training and testing sets
    X_train, y_train, X_test, y_test = split_data(
        data, FEATURE_COLUMNS, TARGET_COLUMN, TRAIN_TEST_SPLIT_RATIO
    )

    # Step 3: Train the XGBoost model
    model = train_xgboost(X_train, y_train, XGBOOST_PARAMS)

    # Step 4: Evaluate the model
    predictions, rmse = evaluate_model(model, X_test, y_test)
    print(f"RMSE: {rmse:.2f}")

    # Step 5: Save the model as a Python script
    save_model_as_py(model, MODEL_PATH, FEATURE_COLUMNS, XGBOOST_PARAMS)
    print(f"Model saved as Python script: {MODEL_PATH}")
