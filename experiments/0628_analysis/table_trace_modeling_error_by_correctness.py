import math
import os
import sys
from matplotlib import pyplot as plt
import numpy as np
from latex_utils import make_latex_table
from sigtest_utils import indistinguishable_from_best
from utils import load_parquet, load_yaml, load_index, load_json_folder, scoring_fns
from datasets import load_dataset
from trace_utils import extractors
folder = 'tables/trace_modeling_error/'
os.makedirs(folder, exist_ok=True)
d = load_yaml('configs/table_trace_modeling_error.yaml')
num_datasets = len(d['lines'])
# difficulty index
difficulty_index = load_index('configs/metrics_difficulty.yaml')
performance_index = load_index('configs/metrics_performance.yaml')
generation_index = load_parquet(load_index('configs/generations.yaml'))
trace_index = load_json_folder(load_index('configs/trace.yaml'))
print(list(trace_index.keys()))

fn_perf = eval(performance_index[tuple(d['metric']['key'])])
kwargs_perf = d['metric']['kwargs']

keeps = []
column_names = []
row_names = [l['name'] for l in d['lines']]
columns_values = []
for column in d['breakdowns']:
    gs = column['graph_spec']
    fn_diff = eval(difficulty_index[tuple(column['difficulty']['key'])])
    fn_diff_criterion = eval(column['criterion'])
    rows_raw_correct = []
    rows_raw_incorrect = []
    for row in d['lines']:
        result_key = tuple([row['extractor']] + row['key'] + [gs])
        results = trace_index[result_key]
        generations_key = tuple(row['key'] + [gs])
        generations = generation_index[generations_key]
        raw_scores_correct = []
        raw_scores_incorrect = []
        rows_raw_correct.append(raw_scores_correct)
        rows_raw_incorrect.append(raw_scores_incorrect)
        for i in range(len(generations)):
            example = generations[i]
            diff = fn_diff(example)
            if not fn_diff_criterion(diff):
                continue
            if i not in results:
                print(f'warning: no example {i}', file=sys.stderr)
                continue
            # get correctness
            acc_score = scoring_fns[row['scorer']](None, example['responses'][0], example['reward_model']['ground_truth'])
            perf = fn_perf(acc_score, **kwargs_perf)
            
            extractor_key = tuple([row['extractor']] + column['key'])
            extractor = extractors[extractor_key]
            score = extractor(results[i])
            if perf:
                raw_scores_correct.append(score)
            else:
                raw_scores_incorrect.append(score)
       
    means = np.array([np.mean(rs) for rs in rows_raw_correct])
    keeps.append([0] * len(means))
    columns_values.append((means * 100).tolist())
    column_names.append(f"\shortstack[c]{{{column['name']}\\\\correct}}")

    means = np.array([np.mean(rs) for rs in rows_raw_incorrect])
    keeps.append([0] * len(means))
    columns_values.append((means * 100).tolist())
    column_names.append(f"\shortstack[c]{{{column['name']}\\\\incorrect}}")

make_latex_table(column_names, row_names, columns_values, keeps, f'tables/trace_modeling_error/table_by_correctness.tex')
