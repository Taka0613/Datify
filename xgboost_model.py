# Machine Learning Model Saved as Python File
from lightgbm import LGBMRegressor

def load_model():
    model = LGBMRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    return model

# Features used for training: ['year', 'month', 'day', 'day_of_week']
