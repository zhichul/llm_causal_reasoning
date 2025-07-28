from __future__ import annotations
from collections import defaultdict
import itertools
import os
import random
import numpy as np
import tqdm
import yaml
import addict
import networkx as nx


from . import discrete_cpt
from . import gaussian_relu
from .utils import NoDefaultDict, copy_file, write_data

def generate_counterfactual_examples(config: addict.Dict, eval_uniqueness_check=True):
    np.random.seed(config.seed)

    if config.model.type == "gaussian_relu":
        data_generator = gaussian_relu.counterfactual
    elif config.model.type == "discrete_cpt":
        data_generator = discrete_cpt.counterfactual
    else:
        raise ValueError(f"Unknown model type: {type}. Did you have typo?")

    splits = ['train', 'dev', 'test']
    per_subsplit_model_counts = {'train': config.data.n_models.train, 
                    'dev': config.data.n_models.dev,
                    'test': config.data.n_models.test,}
    per_subsplit_quest_counts = {'train': config.data.n_questions_per_model.train, 
                    'dev': config.data.n_questions_per_model.dev,
                    'test': config.data.n_questions_per_model.test,}

    total_model_counts = defaultdict(int)
    for split in splits:
        n_vars_sweep = config.model.n_vars.train if split == 'train' else config.model.n_vars.eval
        for n_vars in n_vars_sweep:
            total_model_counts[n_vars] += per_subsplit_model_counts[split]
    dags = defaultdict(list)
    for n_vars in tqdm.tqdm(total_model_counts, desc='Generating graphs'):
        print(f'Generating ({n_vars}) count = {total_model_counts[n_vars]}')
        seen = set()
        for i in tqdm.tqdm(range(total_model_counts[n_vars])):
            tries = 0
            while True:
                g = data_generator.sample_dag(n_vars)
                tries += 1
                if tries > config.data.max_tries_for_uniqueness:
                    raise RuntimeError(f"Tried {tries} times without getting unique graph for evaluation")
                graph_hash = nx.weisfeiler_lehman_graph_hash(g.nx_graph) # guaranteed non-isomorphic graphs will get different hashes
                if not eval_uniqueness_check or graph_hash not in seen:
                    seen.add(graph_hash)
                    break
            dags[n_vars].append((i, g))

    for val in dags.values():
        np.random.shuffle(val)

    print(per_subsplit_quest_counts)
    for split in tqdm.tqdm(splits, desc="split"):
        n_vars_sweep = config.model.n_vars.train if split == 'train' else config.model.n_vars.eval
        n_vals_sweep = config.model.n_vals.train if split == 'train' else config.model.n_vals.eval
            
        for n_vars in n_vars_sweep:
            split_group_dags = dags[n_vars]
            dags[n_vars] = split_group_dags[per_subsplit_model_counts[split]:]
            split_graphs = split_group_dags[:per_subsplit_model_counts[split]]
            for n_vals in n_vals_sweep:
                subsplit = f'{split}_n{n_vars}v{n_vals}'
                split_data = []
                for i, dag in tqdm.tqdm(split_graphs):
                    ex, g, sample = data_generator.generate_counterfactual_example_group(config, 
                        n_quest=per_subsplit_quest_counts[split], 
                        n_vars=n_vars, 
                        n_vals=n_vals, 
                        dag=dag, 
                        extra={            
                        'group_id': f'{i}n{n_vars}',
                        'dag_id': f'{i}n{n_vars}v{n_vals}',
                        'split': split,
                        'subsplit': subsplit, # files will be saved by subsplits
                        }) # this id is only local to this generation to distinguish groups
                    split_data.append((g, sample, ex))
                write_data(config, split_data, subsplit, format='json' if 'format' not in config.data else config.data.format)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'config'
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
        config = NoDefaultDict(config)

    if config.type != "counterfactual":
        raise ValueError(f"Did you have a typo? Running `counterfactual` data generation on a config file of type: {config.type}")
    
    os.makedirs(config.out_dir, exist_ok=True)
    copy_file(args.config, os.path.join(config.out_dir, 'config.yaml'))
    generate_counterfactual_examples(config)