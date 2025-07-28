from itertools import product
import os

from matplotlib import pyplot as plt
import numpy as np
from utils import load_yaml, load_index, load_parquet, materialize_kwargs_perfs, scoring_fns
from sigtest_utils import indistinguishable_from_best
from latex_utils import make_latex_table

folder = 'plots/threshold/'
os.makedirs(folder, exist_ok=True)
d = load_yaml('configs/plot_threshold.yaml')
breakdowns_available = load_yaml('configs/breakdowns.yaml')
num_datasets = len(d['lines'])

difficulty_index = load_index('configs/metrics_difficulty.yaml')
performance_index = load_index('configs/metrics_performance.yaml')
generation_index = load_parquet(load_index('configs/generations.yaml'))

fn_perf = eval(performance_index[tuple(d['metric']['key'])])
kwargs_perfs = materialize_kwargs_perfs(d['metric']['kwargs'])

for breakdown_key in d['breakdowns']:
    breakdown = breakdowns_available[breakdown_key]
    for column in breakdown['columns']:
        fn_diff = eval(difficulty_index[tuple(column['difficulty']['key'])])
        fn_diff_criterion = eval(column['criterion'])
        gs = column['graph_spec']
        rows_raw = []
        rows_names = []
        x = [kwargs_perf[d['x']['key']] - 0.01 for kwargs_perf in kwargs_perfs]
        plt.clf()
        for row in d['lines']:
            rows_names.append(row['name'])
            results = generation_index[tuple(row['key'] + [gs])]
            raw_scores = []
            rows_raw.append(raw_scores)
            for i in range(len(results)):
                example = results[i]
                diff = fn_diff(example)
                if not fn_diff_criterion(diff):
                    continue
                score = scoring_fns[row['scorer']](None, example['responses'][0], example['reward_model']['ground_truth'])
                raw_scores.append(score)
            sc = np.array(raw_scores)
            y_raw = np.stack([fn_perf(sc, **kwargs_perf) for kwargs_perf in kwargs_perfs])
            y_mean = y_raw.mean(-1)
            y_sem= y_raw.std(-1) / np.sqrt(y_raw.shape[-1])
            y_err = y_sem * d['shade_se_multiplier']
            plt.plot(x, y_mean, linestyle=row['linestyle'], color=row['color'], label=row['name'])
            plt.fill_between(x, y_mean-y_sem, y_mean+y_sem, color=row['color'], alpha=d['shade_alpha'])
        plt.legend()
        plt.xlabel(d['x']['name'])
        plt.ylabel('accuracy')
        plt.title(f'{column["name"]} (n={len(raw_scores)})')
        plt.ylim(d['ymin'], d['ymax'])
        plt.savefig(f'plots/threshold/by_{breakdown["name"]}_{column["name"].replace(" ", "_")}.pdf')
        plt.savefig(f'plots/threshold/by_{breakdown["name"]}_{column["name"].replace(" ", "_")}.png', dpi=300)

