import math
import os
from matplotlib import pyplot as plt
import numpy as np
from utils import load_yaml, load_index
from datasets import load_dataset

folder = 'plots/difficulty_statistics/'
os.makedirs(folder, exist_ok=True)
d = load_yaml('configs/plot_difficulty_statistics.yaml')
num_datasets = len(d['lines'])

# difficulty index
metric_index = load_index('configs/metrics_difficulty.yaml')

for metric in d['metrics']:
    plt.clf()
    name = metric['name']
    fn = eval(metric_index[tuple(metric['key'])])
    min_overall = math.inf
    max_overall = -math.inf
    if metric['bins'] == 'integer':
        single_bar_width=1/num_datasets
    elif isinstance(metric['bins'], dict):
        binsize = metric['bins']['size'] 
        single_bar_width= binsize / num_datasets
    else:
        raise ValueError()
    for idx, line in enumerate(d['lines']):
        label = line['name']
        file = line['file']
        color = line['color']
        alpha = line['alpha']
        data = load_dataset('parquet', data_files=file)['train']
        values = [fn(data[i])for i in range(len(data))]

        min_val = min(values)
        max_val = max(values)
        if metric['bins'] == 'integer':
            bins = np.arange(min_val - 0.5, max_val + 1.5, 1)
        elif isinstance(metric['bins'], dict):
            bins = np.arange(min_val // binsize * binsize, (max_val // binsize + 1) * binsize, binsize)
        else:
            raise ValueError()
        frequency, _ = np.histogram(values, bins=bins, density=True)
        offset = (idx - (num_datasets - 1) / 2) * single_bar_width + num_datasets / 2 * single_bar_width 
        plt.bar(
            bins[:-1] + offset,
            frequency,                # density=True equivalent (probability mass)
            width=single_bar_width,
            label=label,
            color=color,
            alpha=alpha,
            edgecolor='black',
            align='center'
        )        
        min_overall = min(min_overall, min_val)
        max_overall = max(max_overall, max_val)
    if metric['bins'] == 'integer':
        plt.xticks(range(min_overall, max_overall + 1))
    elif isinstance(metric['bins'], dict):
        plt.xticks(range(min_overall // binsize * binsize, (max_overall // binsize + 1) * binsize, binsize))
    else:
        raise ValueError()
    plt.xlabel(name)
    plt.ylabel('Frequency')
    plt.legend()
    plt.savefig(os.path.join(folder, f'{metric["filename"]}.pdf'))
    plt.savefig(os.path.join(folder, f'{metric["filename"]}.png'))
    print(metric["filename"], min_overall, max_overall)