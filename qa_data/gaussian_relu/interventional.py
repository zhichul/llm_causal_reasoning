from typing import Any
import addict
import numpy as np
from causal_graph import graph
from ..utils import random_name
import numpy as np


def generate_interventional_example_group(config: addict.Dict, group_id: Any):
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
    intv_samples = []
    examples = []
    for i in range(config.data.n_questions_per_model):
        # pick intervened first to ensure n_observe is possible, and asking a question is possible
        # recall variable indices are topologically sorted naturally, given the way the graph was sampled
        intervened_range = varinds[config.data.n_observe: config.model.n_vars-1]
        intervened_shuffle = np.random.choice(intervened_range, size=(config.data.n_intervene,), replace=False).tolist()
        intervened = intervened_shuffle[:config.data.n_intervene]
        intervened_min, intervened_max = min(intervened), max(intervened)
        observed_shuffle = np.random.choice(varinds[:intervened_min], size=(config.data.n_observe,), replace=False).tolist()
        observed = observed_shuffle[:config.data.n_observe]
        question_vars = varinds[intervened_max+1:]

        # setup the observation
        obs = []
        for j in observed:
            obs_val = sample.endos[i, j].item()
            obs_str = f'{varnames[j]} = {obs_val}'
            obs.append(obs_str)
        observation = ', '.join(obs)

        # setup the intervention
        clamp_endo = {}
        clamp_exo = {vi: val for vi, val in enumerate(sample.exos[i])}
        intv = []
        for j in intervened:
            # Again, following Lampinen et al., 2023 
            # Passive learning of active causal strategies in agents and language models
            intv_val = np.random.choice(config.data.actions) 
            intv_str = f'do({varnames[j]}={intv_val})'
            intv.append(intv_str)
            clamp_endo[j] = intv_val
        intervention = ', '.join(intv)

        intv_sample = g.draw(clamp_endo=clamp_endo, clamp_exo=clamp_exo)
        intv_samples.append(intv_sample) # for bookkeeping

        # setup the question
        k = np.random.choice(question_vars)
        question = f'What is the value of {varnames[k]}?'

        # finish the example
        user_prompt = config.data.prompt.user.format(model_name=gname, 
                                                     var_names=varnames_str,
                                                     observation=observation,
                                                     question=question,
                                                     intervention=intervention)
        reference = str(intv_sample.endos[k].item())
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
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j)
                    } for j in observed
                ],
                'intervention': [
                    {'name': varnames[j],
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in intervened
                ],
                'question': [
                    {'name': varnames[j],
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in [k]
                ],
                'other': [
                    {'name': varnames[j],
                     'value': intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in question_vars if j != k
                ],
            },
            'group_id': group_id,
            'sample_id': i
        })
    
    return examples, g, intv_samples
