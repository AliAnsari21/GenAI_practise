from langchain_text_splitters import RecursiveCharacterTextSplitter,Language

text = """
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def load_data(file_path):
    data = pd.read_csv(file_path)
    print("Dataset loaded successfully")
    print("Shape of dataset:", data.shape)
    return data


def preprocess_data(data):
    data = data.drop_duplicates()
    data = data.dropna()

    print("Data after preprocessing:")
    print(data.head())

    return data


def prepare_features(data, target_column):
    X = data.drop(columns=[target_column])
    y = data[target_column]

    return X, y


def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("Model training completed")
    return model


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("Mean Squared Error:", mse)
    print("R2 Score:", r2)

    return predictions


def main():
    file_path = "housing.csv"

    data = load_data(file_path)
    data = preprocess_data(data)

    X, y = prepare_features(data, "price")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    model = train_model(X_train, y_train)

    predictions = evaluate_model(
        model,
        X_test,
        y_test
    )

    print("Predictions:")
    print(predictions)


if __name__ == "__main__":
    main()
"""

splitter = RecursiveCharacterTextSplitter.from_language(
    language=Language.PYTHON,
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

print(chunks[0])