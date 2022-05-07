from operator import mod
from sklearn.metrics import classification_report
from utils.utils import load_json
import numpy as np
from ml.method import DecisionTree, RandomForest, SVM
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_type",type=str, choices=["DecisionTree","RandomForest","SVM"], help="Choose ML method")
args = parser.parse_args()

print("Loading train data...")
train = load_json('data/train.json')
x_train = []
y_train = []
for item in train:
    x_train.append(item[0])
    y_train.append(item[1])
x_train = np.array(x_train)
y_train = np.array(y_train)
print(f"x_train.shape={x_train.shape}, y_train.shape={y_train.shape}")


print("Loading test data...")
test = load_json('data/test.json')
x_test = []
y_test = []
for item in test:
    x_test.append(item[0])
    y_test.append(item[1])
x_test = np.array(x_test)
y_test = np.array(y_test)
print(f"x_test.shape={x_test.shape}, y_test.shape={y_test.shape}")

print(f"Constructing model {args.model_type}")
if args.model_type == "DecisionTree":
    model = DecisionTree()
elif args.model_type == "RandomForest":
    model = RandomForest()
elif args.model_type == "SVM":
    model = SVM()
else:
    raise NotImplementedError(f"{args.model_type} not implemented")

print(f"Running model {args.model_type}")
model.fit(x_train,y_train)
print(f"Testing model {args.model_type}")
acc = model.acc(x_test,y_test)
print(f"Acc={acc}, model_type={args.model_type}")

pred = model.pred(x_test)
gt = y_test

report = classification_report(gt,pred,target_names=[f"Class_{c}" for c in range(10)])
print(report)

with open(f'result_ml/{args.model_type}.txt','w') as f:
    f.write(report)
