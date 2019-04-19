# Named Entity Recognition with PyTorch

*Author: Prasann Pandya*

## To try, Run
```
python NER.py
```
Type a query and press enter. Type "exit" to exit.

## Requirements

- Python 3
- nltk
- numpy
- torch
- sklearn

## Task

Given a sentence, give a tag to each word ([Named Entity Recognition](https://en.wikipedia.org/wiki/Named-entity_recognition))

```
breaking    news     dealing with prime   minister
B-NEWSTYPE  I-NEWSTYPE  O    O  B-KEYWORDS I-KEYWORDS
```

## Running the Code

1. __Setting Parameters__ The `experiments` directory contains a file `params.json` which sets the hyperparameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 5,
    "num_epochs": 2
}
```

2. __To Train__ simply run
```
python train.py --data_dir data/small --model_dir experiments/base_model
```
It will instantiate a model and train it on the training set following the hyperparameters specified in `params.json`. It will also evaluate some metrics on the development set.

3. __Hyperparameters search__ I created directory `learning_rate` in `experiments`. Run
```
python search_hyperparams.py --data_dir data/small --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

4. __Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

5. __Evaluation on the test set__ Run
```
python evaluate.py --data_dir data/small --model_dir experiments/base_model
```

## Information on the files
- `model/model.py` contains the neural network architecture, loss function and metrics
- `model/data_loader.py` loads the data in batches
- `train.py` for traning the model
- `evaluate.py` for testing the model
- `search_hyperparams.py` for searching hyperparameters
- `utils.py` for saving and leading model checkpoint
