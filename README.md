# README
- Project code for IS303, SJTU
- Multiple classification
## Download Data
- Virus data comes from [Paper](https://arxiv.org/abs/2103.00602) and [Data](https://www.kaggle.com/datasets/datamunge/virusmnist)
- Put your train data `train.csv` under `./data`
- Put your test data `test.csv` under `./data`

## Envs
- PyTorch==1.7.1
- tqdm==4.64.0

## Preprocessing Data
- Follow instructions in `get_data.ipynb` and run this notebook to preprocess data

## Running
- `python main.py`
- More configurations in `main.py`