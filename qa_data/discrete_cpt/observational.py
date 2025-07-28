from __future__ import annotations
import copy
import functools
from typing import Any
import addict
import numpy as np
from causal_graph import graph
from causal_graph.variable_elimination import variable_elimination
from causal_graph.discrete_cpt import Sample
from ..utils import random_name
import numpy as np
from .utils import cpt_to_strs, moralize
import networkx as nx
from causal_graph.discrete_cpt import sample_dag # don't delete this, ../intervention.py needs it
import dwave_networkx as dnx


def generate_observational_example_group(config: addict.Dict, n_quest: int, n_vars: int, n_vals: int|list[int], dag=None, extra={}):
    assert config.data.n_observe == 1
    # sample a graph and a random name
    g = graph.sample_discrete_dag(n_vars=n_vars, n_vals=n_vals, dag=dag)
    gnx = g.nx_graph
    moral = moralize(gnx)
    tw_exact, elim_order = dnx.treewidth_branch_and_bound(moral)

    # sample a varname permutation
    varinds = list(range(n_vars))
    varnames = np.array([f'v{i}' for i in varinds])
    varnames_str = ', '.join(varnames)
    np.random.shuffle(varnames) # reindex
    g_str = g.to_dot(varnames)
    cpt_lines = []
    for cpt_i, cpt in enumerate(g.cpts):
        cpt_lines.append(f"CPTs for {varnames[cpt_i]}:")
        parents = g.parents[cpt_i] + [cpt_i]
        parent_names = [varnames[pa] for pa in parents]
        parent_values = [[str(v) for v in range(g.n_vals[pa])] for pa in parents] # default value is just 0...n_vals-1
        cpt_lines.extend(cpt_to_strs(cpt, parent_names, parent_values)) 
        cpt_lines.append("")
    cpt_str = '\n'.join(cpt_lines)
    
    value_lines = []
    for varname, nval in sorted(zip(varnames, g.n_vals)):
        value_lines.append(f'{varname} can take values in {list(range(nval))}')
    value_str = "\n".join(value_lines)

    # construct example
    system_prompt = config.data.prompt.system
    sample = g.draw(n_quest)
    obs_samples = []
    examples = []
    
    for i in range(n_quest):
        observed = np.random.choice(varinds, size=(config.data.n_observe,), replace=False).tolist()
        question_vars = varinds
        # setup the observation
        obs = []
        cpts = copy.copy(g.cpts)
        parents = copy.copy(g.iparents)
        for j in observed:
            obs_val = sample.endos[i, j].item()
            obs.append((j, obs_val))
            
            # prep for variable eliminatino
            obs_factor = np.zeros(g.n_vals[j])
            obs_factor[obs_val] = 1.
            cpts.append(obs_factor)
            parents.append([j])

        # setup the obs
        observation = ', '.join([f'{varnames[j]}={obs_val}' for j, obs_val in obs])
        obs_sample = Sample(sample.endos[i], sample.exos[i], dict(), dict(), dict(), dict())
        obs_samples.append(obs_sample)

        # setup the question
        k = np.random.choice(question_vars)

        # reference sample
        reference_sample = str(obs_sample.endos[k].item())

        # reference distribution and argmax
        if k not in observed:
            elimination_order = [varind for varind in range(n_vars) if varind != k] # sum everyone out
            reference_distribution, _ = variable_elimination(cpts, parents, elimination_order, renormalize=True) # `sorted`` ONLY toposorts FOR MY GENERATION CODE, since IN MY CODE PARENT ALWAYS HAS SMALLER ID
        else:
            reference_distribution = np.zeros(g.n_vals[j])
            reference_distribution[obs_sample.endos[k].item()] = 1.

        reference_argmax = str(int(reference_distribution.argmax(-1)))
        reference_distribution = str(reference_distribution.round(2).tolist())
    
        if config.data.prompt.reference.type == "sample":
            question = f'Question: Sample a value of {varnames[k]} given it is observed that {observation}?'
            reference = reference_sample
        elif config.data.prompt.reference.type == "distribution":
            question = f'Question: What is the marginal distribution of {varnames[k]} given it is observed that {observation}?'
            reference = reference_distribution
        elif config.data.prompt.reference.type == "argmax":
            question = f'Question: What is the most likely value of {varnames[k]} given it is observed that {observation}?'
            reference = reference_argmax
        else:
            raise ValueError(config.data.prompt.reference.type)

        # finish the example
        format_dict = dict(size=g.n_vars,
                        values=value_str,
                        var_names=varnames_str,
                        question=question,
                        cpts=cpt_str,
                        graph=g_str)

        user_prompt = config.data.prompt.user.format(**format_dict)

        ex = {
            'example': {
                'system': system_prompt,
                'user': user_prompt,
                'reference': reference
            },
            'meta': {
                'variable_names': varnames.tolist(),
                'n_vars': g.n_vars,
                'n_vals': g.n_vals,
                'graph_description': g.to_str(varnames.tolist()), 
            },
            'formal_example': {
                'observed': [
                    {'name': varnames[j],
                     'value': obs_sample.endos[j].item(), 
                     'index': int(j)
                    } for j in observed
                ],
                'intervention': [
                ],
                'question': [
                    {'name': varnames[j],
                     'value': obs_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in [k]
                ],
                'other': [
                    {'name': varnames[j],
                     'value': obs_sample.endos[j].item(), 
                     'index': int(j), 
                    } for j in question_vars if j != k
                ],
            },
            'sample_id': i
        }
        ex.update(extra)
        diff = {
            'treewidth': tw_exact,
        }
        ex.update(diff)
        examples.append(ex)
    return examples, g, obs_samples
