from typing import Any
import addict
import numpy as np
from causal_graph import graph
from ..utils import random_name
import numpy as np


def generate_counterfactual_example_group(config: addict.Dict, group_id: Any):
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
    cf_intv_samples = []
    examples = []
    for i in range(config.data.n_questions_per_model):
        shuffle = np.random.choice(varinds, size=(config.model.n_vars,), replace=False)
        observed = shuffle[:config.data.n_observe]
        observed_shuffle = np.random.choice(observed, size=(config.data.n_observe,), replace=False).tolist()
        counterfactual_intervened = observed_shuffle[:config.data.n_counterfactual]
        # observed values can also be asked about as long as not intervened on
        question_vars = np.concatenate([shuffle[config.data.n_observe:], observed_shuffle[config.data.n_counterfactual:]])
        # setup the observation
        obs = []
        for j in observed:
            obs_val = sample.endos[i, j].item()
            obs_str = f'{varnames[j]} = {obs_val}'
            obs.append(obs_str)
        observation = ', '.join(obs)

        # setup the counterfactual
        clamp_endo = {}
        clamp_exo = {vi: val for vi, val in enumerate(sample.exos[i])}
        cf_intv = []
        for j in counterfactual_intervened:
            # Again, following Lampinen et al., 2023 
            # Passive learning of active causal strategies in agents and language models
            cf_intv_val = np.random.choice(config.data.actions) 
            cf_intv_str = f'do({varnames[j]}={cf_intv_val})'
            cf_intv.append(cf_intv_str)
            clamp_endo[j] = cf_intv_val
        counterfactual = ', '.join(cf_intv)

        cf_intv_sample = g.draw(clamp_endo=clamp_endo, clamp_exo=clamp_exo)
        cf_intv_samples.append(cf_intv_sample) # for bookkeeping

        # setup the question
        k = np.random.choice(question_vars)
        question = f'What is the value of {varnames[k]}?'

        # finish the example
        user_prompt = config.data.prompt.user.format(model_name=gname, 
                                                     var_names=varnames_str,
                                                     observation=observation,
                                                     question=question,
                                                     counterfactual=counterfactual)
        reference = str(cf_intv_sample.endos[k].item())
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
                     'value': cf_intv_sample.endos[j].item(), 
                     'index': int(j)
                    } for j in observed
                ],
                'intervention': [
                    {'name': varnames[j],
                     'value': cf_intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in counterfactual_intervened
                ],
                'question': [
                    {'name': varnames[j],
                     'value': cf_intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in [k]
                ],
                'other': [
                    {'name': varnames[j],
                     'value': cf_intv_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in question_vars if j != k
                ],
            },
            'group_id': group_id,
            'sample_id': i
        })
    
    return examples, g, cf_intv_samples
