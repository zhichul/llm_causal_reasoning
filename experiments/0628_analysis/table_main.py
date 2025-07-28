import os

import numpy as np
from utils import load_yaml, load_index, load_parquet, scoring_fns
from sigtest_utils import indistinguishable_from_best
from latex_utils import make_latex_table

folder = 'tables/main/'
os.makedirs(folder, exist_ok=True)
d = load_yaml('configs/table_main.yaml')
breakdowns_available = load_yaml('configs/breakdowns.yaml')
num_datasets = len(d['lines'])

difficulty_index = load_index('configs/metrics_difficulty.yaml')
performance_index = load_index('configs/metrics_performance.yaml')
generation_index = load_parquet(load_index('configs/generations.yaml'))

fn_perf = eval(performance_index[tuple(d['metric']['key'])])
kwargs_perfs = d['metric']['kwargs']

for breakdown_key in d['breakdowns']:
    breakdown = breakdowns_available[breakdown_key]
    for kwargs_perf in kwargs_perfs:
        keeps = []
        column_names = []
        row_names = [l['name'] for l in d['lines']]
        columns_values = []
        for column in breakdown['columns']:
            fn_diff = eval(difficulty_index[tuple(column['difficulty']['key'])])
            fn_diff_criterion = eval(column['criterion'])
            gs = column['graph_spec']
            rows_raw = []
            for row in d['lines']:
                results = generation_index[tuple(row['key'] + [gs])]
                raw_scores = []
                rows_raw.append(raw_scores)
                for i in range(len(results)):
                    example = results[i]
                    diff = fn_diff(example)
                    if not fn_diff_criterion(diff):
                        continue
                    score = scoring_fns[row['scorer']](None, example['responses'][0], example['reward_model']['ground_truth'])
                    perf = fn_perf(score, **kwargs_perf)
                    raw_scores.append(perf)
            keep, pvals, means = indistinguishable_from_best(rows_raw)
            keeps.append(keep)
            columns_values.append((means * 100).tolist())
            column_names.append(f"\shortstack[c]{{{column['name']}\\\\(n={len(raw_scores)})}}")

        make_latex_table(column_names, row_names, columns_values, keeps, f'tables/main/table_by_{breakdown["name"]}_at_{kwargs_perf["t"]:.2f}.tex')
