from typing import Any
import addict
import numpy as np
from causal_graph import graph
from ..utils import random_name
import numpy as np


def generate_observational_example_group(config: addict.Dict, group_id: Any):
    # sample a graph and a random name
    g = graph.sample_sem(n_vars=config.model.n_vars)
    gname = random_name(config.model.name_n_chars)

    # sample a varname permutation
    varinds = list(range(config.model.n_vars))
    varnames = np.array([f'v{i}' for i in varinds])
    varnames_str = ', '.join(varnames)
    np.random.shuffle(varnames)


    # construct example
    system_prompt = config.data.prompt.system
    sample = g.draw(config.data.n_questions_per_model)

    examples = []
    for i in range(config.data.n_questions_per_model):
        shuffle = np.random.choice(varinds, size=(config.model.n_vars,), replace=False)
        observed = shuffle[:config.data.n_observe]
        unobserved = shuffle[config.data.n_observe:]

        # setup the observation
        obs = []
        for j in observed:
            obs_val = sample.endos[i, j].item()
            obs_str = f'{varnames[j]} = {obs_val}'
            obs.append(obs_str)
        observation = ', '.join(obs)

        # setup the question
        k = np.random.choice(unobserved)
        question = f'What is the value of {varnames[k]}?'

        # finish the example
        user_prompt = config.data.prompt.user.format(model_name=gname, 
                                                     var_names=varnames_str,
                                                     observation=observation,
                                                     question=question)
        reference = str(sample.endos[i, k].item())
        examples.append({
            'example': {
                'system': system_prompt,
                'user': user_prompt,
                'reference': reference
            },
            'meta': {
                'model_name': gname,
                'variable_names': varnames.tolist(),
                'graph_description': g.to_str(varnames.tolist()), 
            },
            'formal_example': {
                'observed': [
                    {'name': varnames[j],
                     'value': sample.endos[i, j].item(), 
                     'index': int(j)
                    } for j in observed
                ],
                'question': [
                    {'name': varnames[j],
                     'value': sample.endos[i, j].item(), 
                     'index': int(j), 
                    } for j in [k]
                ],
                'other': [
                    {'name': varnames[j],
                     'value': sample.endos[i, j].item(), 
                     'index': int(j), 
                    } for j in unobserved if j != k
                ],
            },
            'group_id': group_id,
            'sample_id': i
        })
    return examples, g, sample
