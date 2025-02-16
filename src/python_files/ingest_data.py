import argparse
import logging
import os
import tarfile

import mlflow
import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit


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


def fetch_housing_data(housing_url, housing_path):
    """
    Downloads and extracts the housing data from the specified URL.

    Parameters
    ----------
    housing_url : str
        URL of the housing data tar file.
    housing_path : str
        Directory where the raw housing data will be saved and extracted.

    Raises
    ------
    Exception
        If there is an error during downloading or extraction.
    """
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    logging.info(f"Downloading housing data from {housing_url}...")
    urllib.request.urlretrieve(housing_url, tgz_path)
    logging.info("Extracting housing data...")
    with tarfile.open(tgz_path) as housing_tgz:
        housing_tgz.extractall(path=housing_path)
    logging.info(f"Housing data extracted to {housing_path}.")


def split_data(housing_path, output_dir):
    """
    Splits the housing dataset into training and validation sets using
    StratifiedShuffleSplit to ensure similar income category distributions.

    Parameters
    ----------
    housing_path : str
        Path to the directory containing the raw housing data file (CSV).
    output_dir : str
        Directory where the processed training and validation data will be saved.

    Raises
    ------
    FileNotFoundError
        If the input dataset file is not found.
    """
    csv_path = os.path.join(housing_path, "housing.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    logging.info(f"Reading dataset from {csv_path}...")
    data = pd.read_csv(csv_path)

    logging.info("Creating income category for stratified sampling...")
    data["income_cat"] = pd.cut(
        data["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5],
    )

    logging.info(
        "Splitting dataset into training and"
        + "validation sets using StratifiedShuffleSplit..."
    )
    strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, val_index in strat_split.split(data, data["income_cat"]):
        train_set = data.loc[train_index].drop("income_cat", axis=1)
        val_set = data.loc[val_index].drop("income_cat", axis=1)

    os.makedirs(output_dir, exist_ok=True)
    train_set.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_set.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    logging.info(f"Training and validation data saved to {output_dir}.")


def ingest_data_main():
    """
    Main function to parse arguments and execute data ingestion steps.
    """
    parser = argparse.ArgumentParser(description="Ingest and prepare housing data.")
    parser.add_argument(
        "--housing-url",
        default="https://raw.githubusercontent.com/ageron/handson-ml/master/"
        + "datasets/housing/housing.tgz",
        help="URL of the housing data (default: Ageron's dataset)",
    )
    parser.add_argument(
        "--housing-path",
        default="data/raw",
        help="Path to save the raw housing data (default: ../../data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Path to save the processed datasets (default: ../../data/processed)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--log-path",
        default="logs/ingest_data.log",
        help="Path to save the log file (default: logs/ingest_data.log)",
    )
    parser.add_argument(
        "--no-console-log",
        action="store_true",
        help="Disable console logging",
    )
    parser.add_argument("--parent-run-id", default=None, help="Parent MLflow run ID.")

    args = parser.parse_args()

    if args.log_path:
        log_dir = os.path.dirname(args.log_path)
        os.makedirs(log_dir, exist_ok=True)

    os.makedirs(args.output_dir, exist_ok=True)

    console_log = not args.no_console_log
    setup_logging(args.log_level, args.log_path, console_log)

    logging.info("Starting data ingestion process...")
    fetch_housing_data(args.housing_url, args.housing_path)
    split_data(args.housing_path, args.output_dir)

    mlflow.log_param("housing_url", args.housing_url)
    mlflow.log_param("housing_path", args.housing_path)
    mlflow.log_param("output_dir", args.output_dir)
    logging.info("Data ingestion and preparation completed successfully.")


if __name__ == "__main__":
    ingest_data_main()
