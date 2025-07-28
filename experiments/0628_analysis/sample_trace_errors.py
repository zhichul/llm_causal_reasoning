import json
import math
import os
import random
import sys
from matplotlib import pyplot as plt
import numpy as np
from latex_utils import make_latex_table
from sigtest_utils import indistinguishable_from_best
from utils import load_parquet, load_yaml, load_index, load_json_folder, scoring_fns, write_csv, write_text
from datasets import load_dataset
from trace_utils import extractors
folder = 'samples/trace_errors/'
os.makedirs(folder, exist_ok=True)
d = load_yaml('configs/sample_trace_errors.yaml')
num_datasets = len(d['lines'])
# difficulty index
difficulty_index = load_index('configs/metrics_difficulty.yaml')
generation_index = load_parquet(load_index('configs/generations.yaml'))
trace_index = load_json_folder(load_index('configs/trace.yaml'))
performance_index = load_index('configs/metrics_performance.yaml')
print(list(trace_index.keys()))

n = d['n']

fn_perf = eval(performance_index[tuple(d['metric']['key'])])
kwargs_perf = d['metric']['kwargs']

row_names = [l['name'] for l in d['lines']]

for column in d['breakdowns']:
    random.seed(42)
    gs = column['graph_spec']
    fn_diff = eval(difficulty_index[tuple(column['difficulty']['key'])])
    fn_diff_criterion = eval(column['criterion'])
    rows_raw = []
    rows_correct = []
    rows_responses = []
    rows_prompts = []
    rows_annotations = []
    rows_annotations_scores = []
    for row in d['lines']:
        result_key = tuple([row['extractor']] + row['key'] + [gs])
        results = trace_index[result_key]
        generations_key = tuple(row['key'] + [gs])
        generations = generation_index[generations_key]
        raw_scores = []
        rows_raw.append(raw_scores)
        responses = []
        rows_responses.append(responses)
        prompts = []
        rows_prompts.append(prompts)
        corrects = []
        rows_correct.append(corrects)
        annotations = []
        rows_annotations.append(annotations)
        annotations_scores = []
        rows_annotations_scores.append(annotations_scores)
        for i in range(len(generations)):
            example = generations[i]
            diff = fn_diff(example)
            if not fn_diff_criterion(diff):
                continue
            if i not in results:
                print(f'warning: no example {i}', file=sys.stderr)
                continue
            score = scoring_fns[row['scorer']](None, example['responses'][0], example['reward_model']['ground_truth'])
            raw_scores.append(score)
            responses.append(example['responses'][0])
            # add only user prompts
            prompts.append(example['prompt'][1]['content'])
            corrects.append(fn_perf(score, **kwargs_perf))
            extractor_key = tuple([row['extractor']] + column['key'])
            extractor = extractors[extractor_key]
            error_score = extractor(results[i])
            annotations.append(results[i]['annotations'][0])
            annotations_scores.append(error_score)
    # sample entries
    sample_header = ['prompt', 'system', 'response', 'tvd', column["name"], f'{column["name"]} - human','annotation']
    sample_rows = []
    indices = list(range(len(raw_scores)))
    random.shuffle(indices)
    for i in range(n):
        idx = indices[i]
        # make sure user prompts match
        assert all(prompt[idx] == rows_prompts[0][idx] for prompt in rows_prompts)
        for j in range(len(rows_raw)):
            row = [rows_prompts[0][idx]]
            row.append(row_names[j])
            row.append(rows_responses[j][idx])
            row.append(rows_raw[j][idx])
            row.append(rows_annotations_scores[j][idx])
            row.append("")
            row.append(json.dumps(rows_annotations[j][idx], indent=4))
            sample_rows.append(row)
    write_csv(f'samples/trace_errors/{column["name"].replace(" ", "_")}.both.csv', sample_header, sample_rows)
    write_text(f'samples/trace_errors/{column["name"].replace(" ", "_")}.both.indices.txt', indices)