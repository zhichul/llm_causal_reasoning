from .extract_0_2 import extract_global_formula_error, extract_global_other_error, extract_global_copy_error, extract_global_calculation_error, extract_other_strategy, extract_brute_force_strategy, extract_recursion_strategy, extract_global_no_numeric_computation_error
from .modeling_error_0_1 import extract_modeling_error
from utils import flatten_dict

extractors = flatten_dict({
    'extract_0_2': 
        {
            'formula_global': extract_global_formula_error,
            'other_global': extract_global_other_error,
            'copy_global': extract_global_copy_error,
            'calculation_global': extract_global_calculation_error,
            'strategy_recursion': extract_recursion_strategy,
            'strategy_brute_force': extract_brute_force_strategy,
            'strategy_other': extract_other_strategy,
            'no_numeric_computation_global': extract_global_no_numeric_computation_error,
        },
    'modeling_error_0_1': {
            'modeling_error': extract_modeling_error,
        }
  
})