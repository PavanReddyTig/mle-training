import importlib


def test_package_import():
    """
    Test to verify that the python_files package and its modules are installed correctly.
    """
    try:
        python_files = importlib.import_module("python_files")
        assert python_files is not None, "Failed to import python_files package"

        ingest_data = importlib.import_module("python_files.ingest_data")
        train = importlib.import_module("python_files.train")
        score = importlib.import_module("python_files.score")

        assert ingest_data is not None, "Failed to import ingest_data module"
        assert train is not None, "Failed to import train module"
        assert score is not None, "Failed to import score module"

        print("All modules imported successfully!")
    except Exception as e:
        raise AssertionError(f"Installation test failed: {e}")


if __name__ == "__main__":
    test_package_import()
