import json
import sys
file = sys.argv[1]

print(file)
with open(file, 'rt') as f:
    d = json.loads(f.read())

print('[question]', d['prompt'][1]['content'])
print('[answer]', d['responses'][0])
print('[label]', d['reward_model']['ground_truth'])
for i, analysis in enumerate(d['annotations']):
    print(f'[annotation {i}]')
    print(json.dumps(analysis, indent=4))