# src/pipeline/predict_pipeline.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib
import os

from src.components.data_loader import load_data
from src.components.data_preprocessor import preprocess_data

def run_pipeline():
    df = load_data("artifacts/raw.csv")
    df = preprocess_data(df)

    X = df.drop("income", axis=1)
    y = df["income"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/linear_model.pkl")

    with open("artifacts/mae.txt", "w") as f:
        f.write(f"MAE: {mae}")

    print("Pipeline finished. Model and MAE saved.")
