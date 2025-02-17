import argparse
import logging

import joblib
import mlflow
import numpy as np
import pandas as pd
from scipy.stats import randint
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Custom transformer to add derived attributes.

    Attributes
    ----------
    add_bedrooms_per_room : bool, optional
        Whether to include 'bedrooms_per_room' feature.
    """

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, 0] / X[:, 3]
        population_per_household = X[:, 2] / X[:, 3]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, 1] / X[:, 0]
            return np.c_[
                X, rooms_per_household, population_per_household, bedrooms_per_room
            ]
        return np.c_[X, rooms_per_household, population_per_household]


def setup_logging(log_level, log_path=None, console_log=True):
    """
    Configures logging.

    Parameters
    ----------
    log_level : str
        Logging level (DEBUG, INFO, WARNING, ERROR).
    log_path : str, optional
        Path to save log file.
    console_log : bool, optional
        Whether to log messages to the console.
    """
    handlers = [logging.StreamHandler()] if console_log else []
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
        Path to training dataset.
    val_path : str
        Path to validation dataset.

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
    Preprocesses data: adds derived features,
    encodes categorical variables, and imputes missing values.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset.
    target_column : str, optional
        Target variable column.

    Returns
    -------
    np.ndarray, pd.Series
        Preprocessed feature matrix and target values.
    """
    X = data.drop(columns=[target_column])
    y = data[target_column]

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(include=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numeric_transformer, numerical_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )

    X_preprocessed = preprocessor.fit_transform(X)
    joblib.dump(preprocessor, "model/preprocessor.pkl")

    return X_preprocessed, y


def train_model(train_data, target_column="median_house_value"):
    """
    Trains a Random Forest model with hyperparameter tuning on the training dataset.

    Parameters
    ----------
    train_data : pd.DataFrame
        The training dataset.
    target_column : str, optional
        The target column to predict (default: "median_house_value").

    Returns
    -------
    RandomForestRegressor
        The best Random Forest model from hyperparameter tuning.
    """
    logging.info("Preparing training data...")
    X_train, y_train = preprocess_data(train_data, target_column)

    logging.info("Defining hyperparameter search space...")
    param_distributions = {
        "n_estimators": randint(50, 200),
        "max_features": randint(2, 8),
        "max_depth": randint(10, 50),
    }

    model = RandomForestRegressor(random_state=42)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_mean_squared_error",
        cv=5,
        random_state=42,
        n_jobs=-1,
    )

    logging.info("Starting hyperparameter tuning...")
    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    logging.info(f"Best model parameters: {random_search.best_params_}")

    return best_model


def train_main():
    """
    Main function to handle argument parsing and training workflow.
    """
    parser = argparse.ArgumentParser(
        description="Train and evaluate Random Forest model with hyperparameter tuning."
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
        help="Path to the trained model(default: ./model/random_forest_model.pkl)",
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
    parser.add_argument("--parent-run-id", default=None, help="Parent MLflow run ID.")

    args = parser.parse_args()
    setup_logging(args.log_level, args.log_path, not args.no_console_log)
    train_data, val_data = load_data(args.train_path, args.val_path)
    model = train_model(train_data)
    joblib.dump(model, args.output_path)
    logging.info("Training completed successfully.")
    mlflow.log_param("train_path", args.train_path)
    mlflow.log_param("val_path", args.val_path)
    mlflow.log_param("output_path", args.output_path)
    mlflow.log_artifact(args.output_path)
    mlflow.log_artifact("model/preprocessor.pkl")
    mlflow.log_params(model.get_params())

    mlflow.sklearn.log_model(model, "model")


if __name__ == "__main__":
    train_main()
