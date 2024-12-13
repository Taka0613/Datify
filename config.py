# Configuration for paths and settings
MODEL_PATH = "xgboost_model.py"  # Path to save the model as a Python script
DATA_PATH = "AI需要予測送付用_アイテムフラグ.xlsx"
SKIP_ROWS = 16  # Rows to skip when reading the data
USE_COLUMNS = ["date", "sales"]  # Columns to read from the Excel file
FEATURE_COLUMNS = ["year", "month", "day", "day_of_week"]  # Features for the model
TARGET_COLUMN = "sales"  # Target variable
TRAIN_TEST_SPLIT_RATIO = 0.8  # Ratio for splitting train/test data

# XGBoost model parameters
XGBOOST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 5,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
}
