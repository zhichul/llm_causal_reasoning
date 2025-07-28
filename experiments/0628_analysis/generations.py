import yaml

with open('generations.yaml', 'rt') as f:
    d = yaml.safe_load(f)

print(d)