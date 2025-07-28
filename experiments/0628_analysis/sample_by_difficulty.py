import csv
import os
import random

import numpy as np
from utils import load_yaml, load_index, load_parquet, scoring_fns, write_csv, write_text
from sigtest_utils import indistinguishable_from_best
from latex_utils import make_latex_table

folder = 'samples/by_difficulty/'
os.makedirs(folder, exist_ok=True)
d = load_yaml('configs/sample_by_difficulty.yaml')
breakdowns_available = load_yaml('configs/breakdowns.yaml')
num_datasets = len(d['lines'])

difficulty_index = load_index('configs/metrics_difficulty.yaml')
performance_index = load_index('configs/metrics_performance.yaml')
generation_index = load_parquet(load_index('configs/generations.yaml'))
fn_perf = eval(performance_index[tuple(d['metric']['key'])])
kwargs_perf = d['metric']['kwargs']

n = d['n']

random.seed(42)
for breakdown_key in d['breakdowns']:
    breakdown = breakdowns_available[breakdown_key]
    row_names = [l['name'] for l in d['lines']]
    for column in breakdown['columns']:
        fn_diff = eval(difficulty_index[tuple(column['difficulty']['key'])])
        fn_diff_criterion = eval(column['criterion'])
        gs = column['graph_spec']
        rows_raw = []
        rows_correct = []
        rows_responses = []
        rows_prompts = []
        rows_names = []
        for row in d['lines']:
            rows_names.append(row['name'])
            results = generation_index[tuple(row['key'] + [gs])]
            raw_scores = []
            rows_raw.append(raw_scores)
            responses = []
            rows_responses.append(responses)
            prompts = []
            rows_prompts.append(prompts)
            corrects = []
            rows_correct.append(corrects)
            for i in range(len(results)):
                example = results[i]
                diff = fn_diff(example)
                if not fn_diff_criterion(diff):
                    continue
                score = scoring_fns[row['scorer']](None, example['responses'][0], example['reward_model']['ground_truth'])
                raw_scores.append(score)
                responses.append(example['responses'][0])
                # add only user prompts
                prompts.append(example['prompt'][1]['content'])
                corrects.append(fn_perf(score, **kwargs_perf))

        # sample entries
        sample_header = ['prompt'] + row_names + [f'{name} tvd' for name in row_names]
        sample_rows = []
        indices = list(range(len(raw_scores)))
        random.shuffle(indices)

        for i in range(n):
            idx = indices[i]
            # make sure user prompts match
            assert all(prompt[idx] == rows_prompts[0][idx] for prompt in rows_prompts)
            row = [rows_prompts[0][idx]]
            row.extend([responses[idx] for responses in rows_responses])
            row.extend([scores[idx] for scores in rows_raw])
            sample_rows.append(row)
        
        write_csv(f'samples/by_difficulty/{breakdown["name"]}_{column["name"].replace(" ", "_")}.both.csv', sample_header, sample_rows)
        write_text(f'samples/by_difficulty/{breakdown["name"]}_{column["name"].replace(" ", "_")}.both.indices.txt', indices)
        
        # sample incorrect answer
        indices = list(range(len(raw_scores)))
        random.shuffle(indices)

        for scores, corrects, responses, row_name, prompts in zip(rows_raw, 
                                                                  rows_correct, 
                                                                  rows_responses, 
                                                                  rows_names, 
                                                                  rows_prompts):
            row_indices = [idx for idx in indices if corrects[i]]
            sample_header = ['prompt', row_name, f'{row_name} tvd']
            sample_rows = []
            for i in range(min(n, len(row_indices))):
                idx = row_indices[i]
                # make sure user prompts match
                row = [prompts[idx]]
                row.extend([responses[idx]])
                row.extend([scores[idx]])
                sample_rows.append(row)
            write_csv(f'samples/by_difficulty/{breakdown["name"]}_{column["name"].replace(" ", "_")}_{row_name.replace(" ", "_")}.csv', sample_header, sample_rows)
            write_text(f'samples/by_difficulty/{breakdown["name"]}_{column["name"].replace(" ", "_")}_{row_name.replace(" ", "_")}.indices.txt', row_indices)
        