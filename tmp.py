import os
for m in ["DecisionTree","RandomForest","SVM"]:
    os.system(f'python ml.py --model_type {m}')