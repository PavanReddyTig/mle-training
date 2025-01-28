import os
import subprocess


def test_train_model():
    """
    Test to ensure the model is trained and saved
    """
    result = subprocess.run(
        [
            "python3",
            "src/python_files/train.py",
            "--train-path",
            "data/processed/train.csv",
            "--val-path",
            "data/processed/val.csv",
            "--log-level",
            "DEBUG",
            "--log-path",
            "logs/train.log",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, f"Training script failed: {result.stderr.decode()}"
    assert os.path.exists("model/random_forest_model.pkl"), "Model was not saved"
