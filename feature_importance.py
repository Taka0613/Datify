import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_shap_values(model, X):
    """
    Computes SHAP values for the provided model and dataset.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)
    shap_importances = pd.DataFrame(
        {"feature": X.columns, "importance": abs(shap_values.values).mean(axis=0)}
    ).sort_values("importance", ascending=False)
    return shap_values, shap_importances


def compute_permutation_importance(model, X, y):
    """
    Computes permutation importance for the provided model and dataset.
    """
    result = permutation_importance(
        model, X, y, scoring="neg_mean_squared_error", n_repeats=5, random_state=42
    )
    importance_df = pd.DataFrame(
        {"feature": X.columns, "importance": result.importances_mean}
    )
    importance_df = importance_df.sort_values("importance", ascending=False)
    return importance_df


def plot_shap_summary(shap_values, X):
    """
    Generates a SHAP summary plot and returns the figure.
    """
    # Create a Matplotlib figure
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, plot_type="bar")
    plt.tight_layout()
    return fig


def plot_feature_importance(importance_df, method="SHAP"):
    """
    Generates a feature importance bar chart and returns the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importance_df["feature"], importance_df["importance"], color="skyblue")
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title(f"Feature Importance ({method})")
    ax.invert_yaxis()
    plt.grid()
    plt.tight_layout()
    return fig
