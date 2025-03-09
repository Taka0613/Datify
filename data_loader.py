import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


def load_data(file_path, skip_rows, date_column, sales_column):
    """
    Loads the data and preprocesses it for modeling.
    Dynamically matches columns based on combined header rows (2nd, 4th, 6th, and 8th rows).
    """
    # Step 1: Read specific header rows (2nd, 4th, 6th, and 8th rows)
    header_rows = (
        pd.read_excel(file_path, nrows=8, header=None).iloc[[1, 3, 5, 7]].fillna("")
    )

    # Combine the specified rows to create unique column names
    combined_headers = header_rows.apply(
        lambda x: "_".join(x.astype(str).str.strip()), axis=0
    ).tolist()

    # Ignore the first column and adjust the headers accordingly
    combined_headers = combined_headers[1:]

    # Step 2: Match selected columns with combined headers
    if sales_column not in combined_headers:
        raise ValueError(
            f"Column '{sales_column}' not found. Available columns: {combined_headers}"
        )

    # Map the date column to the first column index and sales column to its matching index
    date_index = 0  # First column is assumed to be the date column
    sales_index = (
        combined_headers.index(sales_column) + 1
    )  # Adjust for skipped first column
    usecols = [date_index, sales_index]

    # Step 3: Load only the necessary columns and skip irrelevant rows
    data = pd.read_excel(file_path, skiprows=skip_rows, usecols=usecols)

    # Rename columns for consistency
    data.columns = ["date", "sales"]
    return data


def extract_column_names(file_path):
    """
    Extracts column names based on the 2nd, 4th, 6th, and 8th rows of the Excel file.
    """
    header_rows = pd.read_excel(file_path, nrows=8, header=None)
    column_names = header_rows.iloc[[1, 3, 5, 7]].fillna("").apply("_".join, axis=0)
    return column_names


def add_features(data):
    """
    Adds additional features to the data for modeling.
    Ensures the 'date' column is in datetime format before feature extraction.
    """
    # Ensure 'date' column is in datetime format
    data["date"] = pd.to_datetime(data["date"], errors="coerce")

    # Drop rows where the date conversion failed
    if data["date"].isnull().any():
        data = data.dropna(subset=["date"])
        print("Dropped rows with invalid dates.")

    # Extract features from the 'date' column
    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    return data


def split_data(data, feature_columns, target_column, split_ratio=0.8):
    """
    Splits the data into training and testing sets.
    """
    split_index = int(split_ratio * len(data))
    train_data = data.iloc[:split_index]
    test_data = data.iloc[split_index:]
    X_train = train_data[feature_columns]
    y_train = train_data[target_column]
    X_test = test_data[feature_columns]
    y_test = test_data[target_column]
    return X_train, y_train, X_test, y_test


def generate_data_insights(data):
    """
    Generates a summary of key dataset insights, including missing values, correlation, and distributions.
    """
    insights = {}

    # Check for missing values
    missing_values = data.isnull().sum()
    missing_percentage = (missing_values / len(data)) * 100
    insights["missing_values"] = missing_percentage[missing_percentage > 0].to_dict()

    # Feature statistics
    feature_stats = data.describe().T
    insights["feature_statistics"] = feature_stats.to_dict()

    # Correlation with target variable
    if "sales" in data.columns:
        correlation = data.corr()["sales"].sort_values(ascending=False)
        insights["correlation"] = correlation.to_dict()

    return insights


def plot_data_distribution(data):
    """
    Plots data distribution for numerical features.
    """
    numerical_cols = data.select_dtypes(include=["number"]).columns
    fig, ax = plt.subplots(
        len(numerical_cols), 1, figsize=(10, len(numerical_cols) * 3)
    )

    for i, col in enumerate(numerical_cols):
        sns.histplot(data[col], kde=True, ax=ax[i])
        ax[i].set_title(f"Distribution of {col}")

    plt.tight_layout()
    return fig
