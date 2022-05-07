from ast import arg
import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", type=str, default="modify")
args = parser.parse_args()

# commit_string = "TPR, FPR, TNR, FNR"
not_add = ['result', 'data', 'weights', 'results',"archive.zip"]
for item in os.listdir():
    if item in not_add:
        continue
    else:
        os.system(f"git add {item}")
os.system(f'git commit -m "{args.m}"')
os.system("git push origin main")