# mle-training
private git repo

## Install the Package
`pip install -e .`

## Usage
To ingest and preprocess the data:  
Example:  
`python src/python_files/ingest_data.py --output-dir data/processed --log-level INFO --log-path logs/ingest_data.log`

To train the model:  
Example:  
`python src/python_files/train.py --train-path data/processed/train.csv --log-path logs/train.log`

To evaluate the model:
Example:  
`python src/python_files/score.py --model-path model/random_forest_model.pkl --test-path data/processed/val.csv --log-path logs/score.log`

## Testing
The tests are located in the tests/ directory.   
To run all tests:
`pytest tests/testing`

