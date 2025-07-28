import json
import random
import re
import os
import os.path
from datasets import load_dataset, concatenate_datasets, Dataset

import numpy as np
import argparse

import re
import numpy as np



    
def extract_solution(solution_str):
    final_answer_str="ANSWER"
    match = re.search(final_answer_str + r"\s*" + r"\[(((\-?[0-9\.])+,\s+)*((\-?[0-9\.])+))\]", solution_str)
    if match is None:
        return None
    else:
        numbers = match.group(1)
        try:
            solution = [float(s) for s in numbers.split(",")]
            return solution
        except:
            return None


def tvd(p1, p2):
    if not ((isinstance(p1, list) or isinstance(p1, np.ndarray)) and \
         (isinstance(p2, list) or isinstance(p2, np.ndarray))):
        raise ValueError((p1, p2))
    if len(p1) != len(p2):
        raise ValueError((p1, p2))
    p1 = np.array(p1)
    p2 = np.array(p2)
    return (np.abs(p1 - p2).sum() / 2).item()

def compute_score(data_source, solution_str, ground_truth, extractor=extract_solution, extra_info=None):
    answer = extractor(solution_str)
    if answer is None: # format error
        reward= 0
    else:
        reward = 0.1 # for formatting correctly
        if len(answer) == len(ground_truth):
            reward += 0.1 # for getting the right length
            reward += 0.8 * max(1-tvd(answer, ground_truth), 0) # for getting dist. correctly
    return {'score': reward, 'reward': reward, 'pred': solution_str}

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    import random
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/ppo/n10v2')
    parser.add_argument('--data_source', default=os.path.join(os.environ['ROOT'], 'data/2025-04-06/interventional_distribution'))
    parser.add_argument('--data_subsplits', nargs="+", default=['n10v2'])
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--monitor_count', type=int, default=None)
    parser.add_argument('--first', type=int, default=None)
    parser.add_argument('--filter_key', type=str, default=None)
    parser.add_argument('--filter_val', type=str, default=None, help='use json format')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_name', type=str, default=None)

    args = parser.parse_args()
    random.seed(args.seed)

    train_datasets = []
    dev_datasets = []
    test_datasets = []
    for subsplit in args.data_subsplits:
        train_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'train_{subsplit}.parquet'))['train'])
        dev_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'dev_{subsplit}.parquet'))['train'])
        test_datasets.append(load_dataset('parquet', data_files=os.path.join(args.data_source, f'test_{subsplit}.parquet'))['train'])
        
    train_dataset = concatenate_datasets(train_datasets)
    dev_dataset = concatenate_datasets(dev_datasets)
    test_dataset = concatenate_datasets(test_datasets)

    if args.debug:
        train_dataset = Dataset.from_dict(train_dataset[:100])
        dev_dataset = Dataset.from_dict(dev_dataset[:100])
        test_dataset = Dataset.from_dict(test_dataset[:100])
    if args.first:
        train_dataset = Dataset.from_dict(train_dataset[:args.first])
        dev_dataset = Dataset.from_dict(dev_dataset[:args.first])
        test_dataset = Dataset.from_dict(test_dataset[:args.first])

    if args.monitor_count:
        train_dataset = train_dataset.shuffle(seed=args.seed)
        dev_dataset = dev_dataset.shuffle(seed=args.seed)
        test_dataset = test_dataset.shuffle(seed=args.seed)
        train_dataset = Dataset.from_dict(train_dataset[:args.monitor_count])
        dev_dataset = Dataset.from_dict(dev_dataset[:args.monitor_count])
        test_dataset = Dataset.from_dict(test_dataset[:args.monitor_count])

    if args.filter_key is not None:
        filter_val = json.loads(args.filter_val)
        train_dataset=  train_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
        dev_dataset=  dev_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)
        test_dataset=  test_dataset.filter(lambda ex: ex[args.filter_key] == filter_val)

    def make_map_fn(split):

        def process_fn(example, idx):
            prompt_raw = example.pop('example')
            system_prompt = prompt_raw['system']
            user_prompt = prompt_raw['user']
            answer = prompt_raw['reference']
            subsplit = example['subsplit']
            group_id = example['group_id']
            dag_id = example['dag_id']
            data = {
                "data_source": (args.data_source if args.data_name is None else args.data_name) + f"_d{example['ancestor_depth']}",
                "prompt": [{
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": json.loads(answer)
                },
                "extra_info": {
                    'split': split,
                    'subsplit': subsplit,
                    'group_id': group_id,
                    'dag_id': dag_id,
                    'index': idx,
                    "prompt":system_prompt + user_prompt,
                    "answer": answer
                }
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    dev_dataset = dev_dataset.map(function=make_map_fn('dev'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    dev_dataset.to_parquet(os.path.join(local_dir, 'dev.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))