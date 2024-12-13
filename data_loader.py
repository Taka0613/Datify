import pandas as pd


def load_data(file_path, skip_rows, use_columns):
    """
    Loads the data and preprocesses it for modeling.
    """
    data = pd.read_excel(
        file_path, skiprows=skip_rows, usecols=use_columns, names=["date", "sales"]
    )
    data["date"] = pd.to_datetime(data["date"])
    data.dropna(subset=["sales"], inplace=True)
    return data


def add_features(data):
    """
    Adds date-related features to the dataset.
    """
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
