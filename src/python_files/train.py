import argparse
import logging
import os

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def setup_logging(log_level, log_path=None, console_log=True):
    """
    Configures logging for the script.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_path : str, optional
        Path to the log file. If None, logs won't be written to a file.
    console_log : bool, optional
        Whether to log messages to the console. Default is True.
    """
    handlers = []
    if console_log:
        handlers.append(logging.StreamHandler())
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_data(train_path, val_path):
    """
    Loads training and validation datasets.

    Parameters
    ----------
    train_path : str
        Path to the training dataset CSV file.
    val_path : str
        Path to the validation dataset CSV file.

    Returns
    -------
    pd.DataFrame, pd.DataFrame
        Training and validation datasets.
    """
    logging.info(f"Loading training data from {train_path}...")
    train_data = pd.read_csv(train_path)
    logging.info(f"Loading validation data from {val_path}...")
    val_data = pd.read_csv(val_path)
    return train_data, val_data


def preprocess_data(data, target_column="median_house_value"):
    """
    Preprocesses the data by handling categorical features and missing values.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset to preprocess.
    target_column : str
        The target column to exclude from preprocessing.

    Returns
    -------
    pd.DataFrame, pd.Series
        Preprocessed features (X) and target (y).
    """
    logging.info("Preprocessing data...")

    X = data.drop(target_column, axis=1)
    y = data[target_column]

    categorical_columns = X.select_dtypes(include=["object"]).columns
    numerical_columns = X.select_dtypes(include=["number"]).columns

    numeric_transformer = SimpleImputer(strategy="median")
    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    X_preprocessed = preprocessor.fit_transform(X)

    return X_preprocessed, y


def train_model(train_data, target_column="median_house_value"):
    """
    Trains a Random Forest model on the training dataset.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset.
    target_column : str, optional
        The target column to predict (default: "median_house_value").

    Returns
    -------
    RandomForestRegressor
        The trained Random Forest model.
    """
    logging.info("Preparing training data...")
    X_train, y_train = preprocess_data(train_data, target_column)

    logging.info("Training Random Forest model...")
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    logging.info("Model training completed.")
    return model


def evaluate_model(model, val_data, target_column="median_house_value"):
    """
    Evaluates the trained model on the validation dataset.

    Parameters
    ----------
    model : sklearn model
        The trained model to evaluate.
    val_data : pd.DataFrame
        The validation dataset.
    target_column : str, optional
        The target column to predict (default: "median_house_value").
    """
    logging.info("Evaluating model on validation data...")

    X_val, y_val = preprocess_data(val_data, target_column)

    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)

    logging.info(f"Validation RMSE: {rmse:.2f}")


def save_model(model, output_path):
    """
    Saves the trained model to a file.

    Parameters
    ----------
    model : RandomForestRegressor
        The trained Random Forest model.
    output_path : str
        Path to save the model file.
    """
    logging.info(f"Saving model to {output_path}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    joblib.dump(model, output_path)
    logging.info(f"Model saved to {output_path}.")


def main():
    """
    Function to create parser, add arguments and
    make function calls to the training function.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate a Random Forest model."
    )
    parser.add_argument(
        "--train-path",
        default="data/processed/train.csv",
        help="Path to the training dataset (default: ./data/processed/train.csv)",
    )
    parser.add_argument(
        "--val-path",
        default="data/processed/val.csv",
        help="Path to the validation dataset (default: ./data/processed/val.csv)",
    )
    parser.add_argument(
        "--output-path",
        default="model/random_forest_model.pkl",
        help="Path to save the trained model(default: ./model/random_forest_model.pkl)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-path",
        default="logs/train.log",
        help="Path to save the log file (default: ./logs/train.log)",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )

    args = parser.parse_args()

    log_dir = os.path.dirname(args.log_path)
    os.makedirs(log_dir, exist_ok=True)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    console_log = not args.no_console_log
    setup_logging(args.log_level, args.log_path, console_log)

    train_data, val_data = load_data(args.train_path, args.val_path)

    model = train_model(train_data)

    evaluate_model(model, val_data)

    save_model(model, args.output_path)

    logging.info("Training process completed successfully.")


if __name__ == "__main__":
    main()
