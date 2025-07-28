import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from dotenv import load_dotenv
from openai import OpenAI
import tqdm
import yaml
import sys
from prompts import extraction_0_1, extraction_0_2, modeling_error_0_1

analysis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '0628_analysis')
sys.path.append(analysis_dir)
from utils import load_index, load_parquet, load_yaml

load_dotenv()
print(os.environ["OPENAI_API_KEY"])
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

PROMPTS = {
    'extraction_0_1': extraction_0_1,
    'extraction_0_2': extraction_0_2,
    'modeling_error_0_1': modeling_error_0_1
}

def annotate(messages, model, max_completion_tokens=4096, temperature=0, response_format=None, n=1, post_process=None):
    if "o3" or "o4" in model:
        output = client.chat.completions.create(messages=messages, model=model, response_format=response_format, max_completion_tokens=max_completion_tokens,n=n)
    else:
        output = client.chat.completions.create(messages=messages, temperature=temperature, model=model, response_format=response_format, max_completion_tokens=max_completion_tokens,n=n)
    annotations = [choice.message.content for choice in output.choices]
    if response_format == {'type': 'json_object'}:
        annotations = [json.loads(annotation) for annotation in annotations]
    if post_process is not None:
        annotations = [post_process(a) for a in annotations]
    return annotations

def in_analysis_dir(path):
    return os.path.join(analysis_dir, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, required=True)
    args = parser.parse_args()
    return args

def load_cache(cache):
    with open(cache, 'rt') as f:
        return json.loads(f.read())

def write_cache(obj, cache):
    with open(cache, 'wt') as f:
        print(json.dumps(obj, indent=4), file=f)

def analyze_with_cache(config, example, cache_path):
    print(f"analyzing {cache_path}")
    n = config['n']
    if os.path.exists(cache_path):
        cache = load_cache(cache_path)
    else:
        cache = example
        cache['annotations'] = []
    if len(cache['annotations']) >= n:
        print(f'cache already complete at {cache_path}')
        return
    exist_count = len(cache['annotations'])

    # construct prompt
    prompt_module = PROMPTS[config['prompt_version']]
    temperature = config['temperature']
    max_completion_tokens = config['max_completion_tokens']
    question = example['prompt'][1]['content']
    answer = example['responses'][0]
    model = config['model']
    messages = prompt_module.format_messages(question, answer, debug=False)

    # annotate and cache
    new_annotations = annotate(messages=messages, max_completion_tokens=max_completion_tokens, temperature=temperature, model=model, response_format=prompt_module.RESPONSE_FORMAT, post_process=prompt_module.POST_PROCESS, n=n-exist_count)
    cache['annotations'].extend(new_annotations)
    write_cache(cache, cache_path)

def main():
    args = parse_args()
    config = load_yaml(args.config)
    print(config)
    breakdowns_available = load_yaml(in_analysis_dir('configs/breakdowns.yaml'))
    generation_index = load_parquet(load_index(in_analysis_dir('configs/generations.yaml')))
    difficulty_index = load_index(in_analysis_dir('configs/metrics_difficulty.yaml'))
    waits = []
    judge_name = f'{config["model"].replace("-","_").replace(".", "_")}_t{config["temperature"]}_len{config["max_completion_tokens"]}'
    with ThreadPoolExecutor(max_workers=config['max_workers']) as pool:
        for breakdown_obj in config['breakdowns']:
            breakdown_key = breakdown_obj['key']
            breakdown_sets = breakdown_obj['sets']
            breakdown_cols = [col for col in breakdowns_available[breakdown_key]['columns'] if col['name'] in breakdown_sets]

            for column in breakdown_cols:
                fn_diff = eval(difficulty_index[tuple(column['difficulty']['key'])])
                fn_diff_criterion = eval(column['criterion'])
                gs = column['graph_spec']
                for row in config['lines']:
                    result_key = tuple(row['key'] + [gs])
                    results = generation_index[result_key]
                    folder = os.path.join(config['output_folder'], judge_name, *result_key)
                    os.makedirs(folder, exist_ok=True)
                    cnt = 0
                    for i in range(len(results)):
                        # if cnt >= 1:
                        #     break
                        example = results[i]
                        diff = fn_diff(example)
                        if not fn_diff_criterion(diff):
                            continue
                        cnt += 1
                        cachefile = os.path.join(folder, f'{i}.json')
                        waits.append(pool.submit(analyze_with_cache, config, example, cachefile))
    
    for wait in tqdm.tqdm(waits):
        wait.result()
        continue

if __name__ == '__main__':
    main()