import pandas as pd
import tqdm
from datasets import load_dataset
import json

def load_data(path):
    f = open(path, 'r', encoding='utf-8')
    content = f.read()
    data = json.loads(content)
    f.close()
    return data

