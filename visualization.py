import matplotlib.pyplot as plt
import plotly.express as px


def plot_actual_vs_predicted(test_data, predictions, title="Daily Sales Prediction"):
    plt.figure(figsize=(12, 6))
    plt.plot(
        test_data["date"], test_data["sales"], label="Actual Sales", linestyle="--"
    )
    plt.plot(test_data["date"], predictions, label="Predicted Sales", alpha=0.7)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.legend()
    plt.grid()
    plt.show()


def plot_feature_importance(importance_df, method="SHAP"):
    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        title=f"Feature Importance ({method})",
    )
    fig.show()
