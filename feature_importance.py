import matplotlib.pyplot as plt
import shap
import pandas as pd
from sklearn.inspection import permutation_importance
import numpy as np


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


def plot_shap_summary(shap_values, X, top_n=None):
    """
    Generates a SHAP summary plot and returns the figure.

    Parameters:
    - shap_values: SHAP values of the dataset
    - X: The original features DataFrame
    - top_n: If provided, only plot the top_n most important features
    """
    # If top_n is specified, reduce shap_values and X to top_n features
    if top_n is not None:
        # Compute mean absolute SHAP values to determine importance
        mean_abs_shap = np.mean(np.abs(shap_values.values), axis=0)
        top_indices = np.argsort(mean_abs_shap)[-top_n:]

        # Filter the dataset and shap values to top_n features
        X_top = X.iloc[:, top_indices]
        shap_values_top = shap_values[:, top_indices]
    else:
        X_top = X
        shap_values_top = shap_values

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values_top, X_top, show=False, plot_type="bar")
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


def plot_shap_dependence(shap_values, X, feature, interaction_feature=None):
    """
    Generates a SHAP dependence plot for a specific feature.

    Parameters:
    - shap_values: SHAP values calculated from the model and dataset.
    - X: The dataset (Pandas DataFrame).
    - feature: The feature for which to plot the dependence.
    - interaction_feature: An optional feature to highlight interactions.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.dependence_plot(
        feature, shap_values.values, X, interaction_index=interaction_feature, ax=ax
    )
    plt.tight_layout()
    return fig
