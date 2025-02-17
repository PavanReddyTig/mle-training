import argparse
import logging

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

from python_files.train import CombinedAttributesAdder


def setup_logging(log_level, log_path=None, console_log=True):
    """
    Sets up logging configuration.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_path : str, optional
        Path to save log file.
    console_log : bool, optional
        Whether to enable console logging (default: True).
    """
    handlers = [logging.StreamHandler()] if console_log else []
    if log_path:
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )


def load_data(data_path):
    """
    Loads a dataset from a given CSV file.

    Parameters
    ----------
    data_path : str
        Path to the dataset CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset.
    """
    logging.info(f"Loading data from {data_path}...")
    return pd.read_csv(data_path)


def preprocess_data(data, preprocessor_path, target_column="median_house_value"):
    """
    Preprocesses data using a saved preprocessor.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    preprocessor_path : str
        Path to the saved preprocessor.
    target_column : str, optional
        Target variable column.

    Returns
    -------
    np.ndarray, pd.Series
        Preprocessed feature matrix and target values.
    """
    logging.info("Loading preprocessor...")
    preprocessor = joblib.load(preprocessor_path)

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_preprocessed = preprocessor.transform(X)
    return X_preprocessed, y


def evaluate_model(model_path, preprocessor_path, data_path):
    """
    Loads a trained model and evaluates it on a dataset.

    Parameters
    ----------
    model_path : str
        Path to the trained model file.
    preprocessor_path : str
        Path to the preprocessor file.
    data_path : str
        Path to the dataset CSV file.

    Returns
    -------
    float
        Root Mean Squared Error (RMSE) on the dataset.
    """
    logging.info("Loading trained model...")
    model = joblib.load(model_path)

    data = load_data(data_path)
    X, y = preprocess_data(data, preprocessor_path)

    logging.info("Making predictions...")
    predictions = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, predictions))
    logging.info(f"Evaluation completed. RMSE: {rmse:.4f}")
    return rmse


def score_main():
    """
    Main function for scoring the model.

    This function loads the test data, model, and preprocessor, then preprocesses
    the test data, evaluates the model, and logs the results.
    """
    parser = argparse.ArgumentParser(
        description="Score a trained model on the test dataset."
    )
    parser.add_argument(
        "--test-path",
        default="data/processed/val.csv",
        help="Path to the test dataset (default: ./data/processed/test.csv)",
    )
    parser.add_argument(
        "--model-path",
        default="model/random_forest_model.pkl",
        help="Path to the trained model (default: ./model/random_forest_model.pkl)",
    )
    parser.add_argument(
        "--preprocessor-path",
        default="model/preprocessor.pkl",
        help="Path to the preprocessor pipeline(default: ./model/preprocessor.pkl)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-path",
        default="logs/score.log",
        help="Path to save the log file(default: ./logs/score.log)",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )
    parser.add_argument(
        "--parent-run-id",
        default=None,
        help="Parent MLflow run ID.",
    )

    args = parser.parse_args()
    setup_logging(args.log_level, args.log_path, not args.no_console_log)
    rmse = evaluate_model(args.model_path, args.preprocessor_path, args.test_path)
    logging.info(f"Final RMSE: {rmse:.4f}")
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_param("val_path", args.test_path)
    mlflow.log_param("model_path", args.model_path)


if __name__ == "__main__":
    score_main()
