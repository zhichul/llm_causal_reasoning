# def flatten_dict(d):

import csv
from itertools import product
import json
import os
from datasets import load_dataset
import numpy as np
import yaml

def load_json(file):
    with open(file, 'rt') as f:
        d = json.loads(f.read())
    return d

def load_single_json_folder(folder):
    out = {}
    names = os.listdir(folder)
    for name in names:
        if name.endswith(".json"):
            path = os.path.join(folder, name)
            key = int(name[:-5])
            out[key] = load_json(path)
    return out

def load_json_folder(index):
    out = {}
    for key, val in index.items():
        if isinstance(val, dict):
            out[key] = load_parquet(val)
        else:
            assert isinstance(val, str)
            out[key] = load_single_json_folder(val)
    return out

def load_yaml(file):
    with open(file, 'rt') as f:
        d = yaml.safe_load(f)
    return d

def load_index(name, flatten=True):
    d = load_yaml(name)
    if flatten:
        d = flatten_dict(d)
    return d

def load_index(name, flatten=True):
    d = load_yaml(name)
    if flatten:
        d = flatten_dict(d)
    return d

def load_single_generation_file(file):
    data = load_dataset('parquet', data_files=file)['train']
    return data

def load_parquet(generation_index):
    out = {}
    for key, val in generation_index.items():
        if isinstance(val, dict):
            out[key] = load_parquet(val)
        else:
            assert isinstance(val, str)
            out[key] = load_single_generation_file(val)
    return out

def flatten_dict(d):
    """
    Turns a tree structured nested dict into one-level, where the key is a tuple representing the path in the tree.
    """
    return dict(flatten_key(d))

def flatten_key(d):
    if isinstance(d, dict):
        for key, val in d.items():
            if not isinstance(val, dict):
                yield (key,), val
            else:
                for subkey, subval in flatten_key(val):
                    yield (key,) + subkey, subval
    else:
        raise ValueError("must be a dict")
    
from score_tvd import compute_score as cs_tvd
from score_tvd_sft import compute_score as cs_tvd_sft

scoring_fns = {
    'score_tvd_sft': lambda *args, **kwargs: 1-cs_tvd_sft(*args, **kwargs)['score'],
    'score_tvd': lambda *args, **kwargs: 1-cs_tvd(*args, **kwargs)['score'],
}

def tuples_to_dict(names, tuples):
    return [
        dict(zip(names, tup)) for tup in tuples
    ]

def materialize_kwargs_perfs(desc):
    values = []
    names = []
    for key in desc:
        names.append(key)
        values.append(np.linspace(desc[key]['min'], desc[key]['max'], desc[key]['num']).tolist())
    return tuples_to_dict(names, product(*values))

def write_csv(file, header, rows):
    with open(file, 'wt') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

def write_text(file, lines):
    with open(file, 'wt') as f:
        for line in lines:
            print(line, file=f)
            
if __name__ == "__main__":
    import yaml

    for k, v in load_parquet(load_index('configs/generations.yaml')).items():
        print(f'key={k} val={v}')