import json
import torch
import datetime
import pickle as pkl
import random
import numpy as np

def get_datetime():
    time1 = datetime.datetime.now()
    time2 = datetime.datetime.strftime(time1,'%Y-%m-%d-%H-%M-%S')
    return time2

class Logger():
    def __init__(self,log_file_path) -> None:
        self.path = log_file_path
        with open(self.path,'w') as f:
            f.write(get_datetime() + "\n")
            print(get_datetime())
        return
    
    def log(self,content):
        with open(self.path,'a') as f:
            f.write(content + "\n")
            print(content)
        return

def setup_seed(seed = 3407):
     torch.manual_seed(seed)
     np.random.seed(seed)
     random.seed(seed)

def fetch_update_index():
    with open('index/index.txt', 'r+') as f:
        index = int(f.read())
        f.seek(0)
        f.truncate()
        f.write(str(index+1))
        index = "%06d" % index
        return index

def load_json(path):
    with open(path,'r') as f:
        ans = json.load(f)
    return ans

def save_json(obj, path):
    with open(path, 'w', encoding='utf8') as f:
        json.dump(obj, f, indent=4)

def load_pkl(pkl_path):
    with open(pkl_path,'rb') as f:
        m = pkl.load(f)
        return m

def save_pkl(obj,pkl_path):
    '''obj, path'''
    with open(pkl_path,'wb') as f:
        pkl.dump(obj,f)    