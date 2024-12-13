from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error


def train_xgboost(X_train, y_train, params):
    """
    Trains the XGBoost model with the provided parameters.
    """
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and returns predictions and RMSE.
    """
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    return predictions, rmse


def save_model_as_py(model, model_path, feature_columns, params):
    """
    Saves the trained model as a Python script.
    """
    with open(model_path, "w") as f:
        f.write("# XGBoost Model Saved as Python File\n")
        f.write("from xgboost import XGBRegressor\n\n")
        f.write("def load_model():\n")
        f.write("    model = XGBRegressor(\n")
        for key, value in params.items():
            f.write(f"        {key}={repr(value)},\n")
        f.write("    )\n")
        f.write(f"    model.load_model({repr(model.get_booster().save_raw())})\n")
        f.write("    return model\n\n")
        f.write(f"# Features used for training: {feature_columns}\n")
