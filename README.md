# README
- Project code for IS303, SJTU
- Multiple classification
## Download Data
- Virus data comes from [Paper](https://arxiv.org/abs/2103.00602) and [Data](https://www.kaggle.com/datasets/datamunge/virusmnist)
- Put your train data `train.csv` under `./data`
- Put your test data `test.csv` under `./data`

## Envs
- Python==3.6
- PyTorch==1.7.1
- tqdm==4.64.0
- scikit-learn==0.24.2

## Preprocessing Data
- Follow instructions in `get_data.ipynb` and run this notebook to preprocess data

## Running NN Methods
- `python main.py`, you can find the result in `result`
- More configurations in `main.py` or run `python main.py -h`

## Running ML Methods
- `python ml.py --model_type <model_type>`, you can find the result in `result_ml`
```console
model_type
    DecisionTree
    RandomForest
    SVM
```