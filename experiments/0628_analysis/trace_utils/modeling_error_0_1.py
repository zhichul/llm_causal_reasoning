def extract_modeling_error(example, i=0):
    return int(example['annotations'][i]['value'] == 1)