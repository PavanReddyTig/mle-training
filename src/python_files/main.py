import subprocess

import mlflow

from python_files.ingest_data import ingest_data_main
from python_files.score import score_main
from python_files.train import train_main


def main():
    """Main function to orchestrate MLflow runs."""
    mlflow.set_experiment("ML_Pipeline")

    with mlflow.start_run(run_name="Pipeline") as parent_run:
        parent_run_id = parent_run.info.run_id
        print("Parent Run ID:", parent_run_id)
        with mlflow.start_run(run_name="Ingest Data", nested=True) as child_run1:
            child1_run_id = child_run1.info.run_id
            print("Child1 Run ID, Ingest Data:", child1_run_id)
            ingest_data_main()

        with mlflow.start_run(run_name="Train Model", nested=True) as child_run2:
            child2_run_id = child_run2.info.run_id
            print("Child2 Run ID, Train Model:", child2_run_id)
            train_main()

        with mlflow.start_run(run_name="Score Model", nested=True) as child_run3:
            child3_run_id = child_run3.info.run_id
            print("Child3 Run ID, Score Model:", child3_run_id)
            score_main()


if __name__ == "__main__":
    main()
