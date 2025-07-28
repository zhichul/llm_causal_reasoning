def extract_global_formula_error(example, i=0):
    return int(any('errors' in step and 'formula' in step['errors'] for step in example['annotations'][i]['intermediate_steps']))

def extract_global_other_error(example, i=0):
    return int(any('errors' in step and 'other' in step['errors'] for step in example['annotations'][i]['intermediate_steps']))

def extract_global_copy_error(example, i=0):
    return int(any('errors' in step and 'copy' in step['errors'] for step in example['annotations'][i]['intermediate_steps']))

def extract_global_calculation_error(example, i=0):
    return int(any('errors' in step and 'calculation' in step['errors'] for step in example['annotations'][i]['intermediate_steps']))

def extract_global_no_numeric_computation_error(example, i=0):
    return int(any('errors' in step and 'no numeric computation' in step['errors'] for step in example['annotations'][i]['intermediate_steps']))

def extract_recursion_strategy(example, i=0):
    if example['annotations'][i]['strategy'] == 'recursion':
        return 1
    else:
        return 0

def extract_brute_force_strategy(example, i=0):
    if example['annotations'][i]['strategy'] == 'brute force summation':
        return 1
    else:
        return 0

def extract_other_strategy(example, i=0):
    if example['annotations'][i]['strategy'] == 'other':
        return 1
    else:
        return 0
    