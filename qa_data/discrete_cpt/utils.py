import itertools
import numpy as np
import networkx as nx

def cpt_to_strs(cpt: np.ndarray, parents: list[str], values: list[list[str]], rounding=2):
    lines = []
    for value_inds in itertools.product(*[list(range(len(val))) for val in values[:-1]]): # the last one is child values, which we vectorize when printing
        assignment_tuple = [(parents[i], values[i][value_ind]) for i, value_ind in enumerate(value_inds)]
        if len(value_inds) == 0:
            lines.append(f"P({parents[-1]}) = {str(cpt[value_inds].round(rounding).tolist())}")
        else:
            lines.append(f"P({parents[-1]} | {','.join([f'{parent}={val}'for parent, val in assignment_tuple])}) = {str(cpt[value_inds].round(rounding).tolist())}")
    return lines

def moralize(dag: nx.DiGraph) -> nx.Graph:
    """Marry co‑parents and drop directions."""
    M = nx.Graph()
    M.add_nodes_from(dag.nodes)
    M.add_edges_from(dag.to_undirected().edges())      # skeleton
    for child in dag:
        parents = list(dag.predecessors(child))
        for i, u in enumerate(parents):
            for v in parents[i + 1:]:
                M.add_edge(u, v)                       # marry parents
    return M
