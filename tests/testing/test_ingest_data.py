import os
import subprocess


def test_ingest_data():
    """
    Test to check if processed data files are created after we process
    and split the input dataset.
    """
    result = subprocess.run(
        [
            "python3",
            "src/python_files/ingest_data.py",
            "--output-dir",
            "data/processed",
            "--log-level",
            "DEBUG",
            "--log-path",
            "logs/ingest_data.log",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, f"Ingestion script failed: {result.stderr.decode()}"
    assert os.path.exists("data/processed/train.csv"), "train.csv was not created"
    assert os.path.exists("data/processed/val.csv"), "val.csv was not created"
