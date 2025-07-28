from __future__ import annotations
import copy
import functools
import sys
from typing import Any
import addict
import numpy as np
from causal_graph import graph
from causal_graph.variable_elimination_with_heuristic import min_fill_variable_elimination
from ..utils import random_name
import numpy as np
from .utils import cpt_to_strs, moralize
import networkx as nx
from causal_graph.discrete_cpt import sample_dag # don't delete this, ../intervention.py needs it
import dwave_networkx as dnx

def ancestor_subgraph(g, node):
    return g.subgraph(nx.ancestors(g, node)|{node}).copy()

def dag_longest_path_length_to_node(g, node):
    g = ancestor_subgraph(g, node)
    return nx.dag_longest_path_length(g)

def graph_surgery(g, src):
    g = g.copy()
    g.remove_edges_from(list(g.in_edges(src)))
    return g

@functools.cache
def difficulty(src, tar, g):
    g = graph_surgery(g, src)
    shortest_paths = nx.shortest_path_length(g, target=tar)
    if src in shortest_paths:
        if shortest_paths[src] == 0:
            depth = 0
        else:
            shortest_paths[src] = shortest_paths[src] - 1
            depth = max(shortest_paths.values())
    else:
        depth = max(shortest_paths.values())

    return {
        'ancestor_depth': depth,
        'longest_path_depth': dag_longest_path_length_to_node(g, tar),
        'on_path': nx.has_path(g, source=src, target=tar)
    }

def generate_interventional_example_group(config: addict.Dict, n_quest: int, n_vars: int, n_vals: int|list[int], dag=None, extra={}):
    assert config.data.n_intervene == 1
    # sample a graph and a random name
    g = graph.sample_discrete_dag(n_vars=n_vars, n_vals=n_vals, dag=dag)
    gnx = g.nx_graph
    
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
    intv_samples = []
    examples = []
    
    for i in range(n_quest):
        # pick intervened first to ensure n_observe is possible, and asking a question is possible
        # recall variable indices are topologically sorted naturally, given the way the graph was sampled
        intervened_range = varinds[config.data.n_observe: n_vars-1]
        intervened_shuffle = np.random.choice(intervened_range, size=(config.data.n_intervene,), replace=False).tolist()
        intervened = intervened_shuffle[:config.data.n_intervene]
        intervened_min, intervened_max = min(intervened), max(intervened)
        observed_shuffle = np.random.choice(varinds[:intervened_min], size=(config.data.n_observe,), replace=False).tolist()
        observed = observed_shuffle[:config.data.n_observe]
        question_vars = varinds
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
        cpts = copy.copy(g.cpts)
        parents = copy.copy(g.iparents)
        for j in intervened:
            # Again, following Lampinen et al., 2023 
            # Passive learning of active causal strategies in agents and language models
            intv_val = np.random.choice(g.n_vals[j]) 
            intv.append((j, intv_val))
            clamp_endo[j] = intv_val

            # prep for variable eliminatino
            parents[j] = [j]
            cpts[j] = np.zeros(g.n_vals[j])
            cpts[j][intv_val] = 1.

        intervention = ', '.join([f'do({varnames[j]}={intv_val})' for j, intv_val in intv])

        intv_sample = g.draw(clamp_endo=clamp_endo, clamp_exo=clamp_exo)
        intv_samples.append(intv_sample) # for bookkeeping

        # setup the question
        k = np.random.choice(question_vars)

        # reference sample
        reference_sample = str(intv_sample.endos[k].item())


        # reference distribution and argmax
        assert len(intervened) == 1
        gmod = graph_surgery(gnx, intervened[0])
        relevant_subgraph = ancestor_subgraph(gmod, k)
        moral = moralize(relevant_subgraph)
        tw_exact, _ = dnx.treewidth_branch_and_bound(moral)
        vars_to_eliminate = sorted((nx.ancestors(relevant_subgraph, k) | set(intervened)) - {k}) # do not sum out k, in case we intervened on k and asked about k...
        cpts = [cpts[i] for i in vars_to_eliminate + [k]]
        parents = [parents[i] for i in vars_to_eliminate + [k]]
        reference_distribution, _, num_prods, num_sums = min_fill_variable_elimination(cpts, parents, vars_to_eliminate) # `sorted`` ONLY toposorts FOR MY GENERATION CODE, since IN MY CODE PARENT ALWAYS HAS SMALLER ID

        reference_argmax = str(int(reference_distribution.argmax(-1)))
        reference_distribution = str(reference_distribution.round(2).tolist())
    
        if config.data.prompt.reference.type == "sample":
            question = f'Question: Sample a value of {varnames[k]} given we intervented to set {varnames[intv[0][0]]} to {intv[0][1]}?'
            reference = reference_sample
        elif config.data.prompt.reference.type == "distribution":
            question = f'Question: What is the marginal distribution of {varnames[k]} given we intervented to set {varnames[intv[0][0]]} to {intv[0][1]}?'
            reference = reference_distribution
        elif config.data.prompt.reference.type == "argmax":
            question = f'Question: What is the most likely value of {varnames[k]} given we intervented to set {varnames[intv[0][0]]} to {intv[0][1]}?'
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
        if config.data.n_observe > 0:
            format_dict['observation'] = observation

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
            'sample_id': i,
            'min_fill_products': num_prods,
            'min_fill_sums': num_sums,
            'ancestor_treewidth': tw_exact,
            'ancestor_size': relevant_subgraph.number_of_nodes(),
        }
        ex.update(extra)
        diff = difficulty(intv[0][0],k,g.nx_graph)
        ex.update(diff)
        examples.append(ex)
    
    return examples, g, intv_samples
