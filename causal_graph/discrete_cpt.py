from __future__ import annotations
import dataclasses
from functools import reduce
import numpy as np
from scipy import stats
import networkx as nx

def categorical(p, uni):
    return (p.cumsum(-1) >= uni).argmax(-1) # The indices corresponding to the FIRST occurrence are returned in case of tie.

def match(v1, v2):
    match = v1 == v2
    if isinstance(match, bool):
        if not match: return False
    elif isinstance(match, np.ndarray):
        if not match.all(): return False
    else:
        raise AssertionError
    return True

def match_list(v1, v2):
    assert len(v1) == len(v2)
    for (vv, uu) in zip(v1, v2):
        if not match(vv, uu):
            return False
    return True

def match_dict(v1, v2):
    assert len(v1) == len(v2) and v1.keys() == v2.keys()
    for k in v1:
        if not match(v1[k], v2[k]):
            return False
    return True

def squeeze_dict(d: dict[any, np.ndarray], *args, **kwargs):
    return {k: v.squeeze(*args, **kwargs) for k, v in d.items()}

@dataclasses.dataclass
class Sample:

    endos: np.ndarray
    exos: np.ndarray
    clamp_endo: dict[int, float|np.ndarray]
    clamp_exo: dict[int, float|np.ndarray]
    endos_before_clamp: dict[int, float|np.ndarray]
    exos_before_clamp: dict[int, float|np.ndarray]

    def __str__(self):
        return (f"endos: {self.endos}\n"
                f"exos: {self.exos}\n"
                f"clamp_endo: {self.clamp_endo}\n"
                f"endos_before_clamp: {self.endos_before_clamp}\n"
                f"clamp_exo: {self.clamp_exo}\n"
                f"exos_before_clamp: {self.exos_before_clamp}\n")

    def __eq__(self, value):
        if not (isinstance(value, Sample) and
                match(value.endos, self.endos) and
                match(value.exos, self.exos) and 
                match_dict(value.endos_before_clamp, self.endos_before_clamp) and
                match_dict(value.exos_before_clamp, self.exos_before_clamp) and
                match_dict(value.clamp_endo, self.clamp_endo) and
                match_dict(value.clamp_exo, self.clamp_exo)):
            return False
        return True

@dataclasses.dataclass
class DiscreteCPT:

    n_vars: int                                         # number of variables
    n_vals: int                                         # number of vals for each variable
    n_indep: int                                        # number of independent vars             
    parents: list[list | np.ndarray]                    # parents (in canonical index order)                            (2d array)
    cpts: list[list | np.ndarray]                       # list of CPTs, each (nvals[par{i}],...,nvals[self])            (nd array)

    @property
    def iparents(self):
        if getattr(self, '_iparents', None) is None:
            self._iparents = [pas + [i] for i, pas in enumerate(self.parents)]
        return self._iparents

    def __eq__(self, value):
        if not (isinstance(value, DiscreteCPT) and 
            value.n_vars == self.n_vars and 
            value.n_vals == self.n_vals and 
            value.n_indep == self.n_indep and 
            match_list(value.parents, self.parents) and
            match_list(value.cpts, self.cpts)):
            return False
        return True
            
    def __str__(self):
        if getattr(self, '_default_varnames', None) is None:
            self._default_varnames = [f'#{i}' for i in range(self.n_vars)]
        return self.to_str(self._default_varnames)

    def to_str(self, varnames: list[str]):
        lines = []
        for i, (parents, cpt) in enumerate(zip(self.parents, self.cpts)):
            if len(parents) == 0:
                lines.append(f"{varnames[i]} ~ {cpt}")
            else:
                lines.append(f"{varnames[i]} ~ Choice(⋅|{', '.join([varnames[p] for p in parents])})")
        return "\n".join(lines)
    
    def to_gml(self, varnames: list[str]):
        graph = nx.DiGraph()
        graph.add_nodes_from(varnames)
        for i, pas in enumerate(self.parents):
            graph.add_edges_from([(varnames[pa], varnames[i]) for pa in pas])
        return nx.generate_gml(graph)

    def to_dot(self, varnames: list[str]):
        """
        Always presents in sorted name order.
        """
        from io import StringIO
        graph = nx.DiGraph()
        raw_inds = list(range(len(varnames)))
        paired_inds = sorted(list(zip(varnames, raw_inds)))
        sorted_varnames = [x[0] for x in paired_inds]
        sorted_raw_inds = [x[1] for x in paired_inds]

        graph.add_nodes_from(sorted_varnames)
        for i in sorted_raw_inds:
            pas = self.parents[i]
            graph.add_edges_from([(varnames[pa], varnames[i]) for pa in pas])
        buffer = StringIO()
        nx.drawing.nx_pydot.write_dot(graph, buffer)
        return buffer.getvalue()

    @property
    def nx_graph(self):
        if getattr(self, '_nx_graph', None) is None:
            self._nx_graph = nx_graph(self.n_vars, self.parents)
        return self._nx_graph

    def draw(self, n: int = None, clamp_endo: dict[int, float|np.ndarray] = None, clamp_exo: dict[int, float|np.ndarray] = None):
        """
        Draw `n` samples, optionally clamping 
            endogenous variables (e.g., v1, v2, v3) for interventions
            exogenous variables (e.g. the gaussian noises used to generate v1, v2, v3) for counterfactuals
        """
        if clamp_endo is None:
            clamp_endo = dict()
        if clamp_exo is None:
            clamp_exo = dict()

        # sample exogenous vars as quantiles
        exos = np.random.uniform(size=(1, self.n_vars,) if n is None else (n, self.n_vars))
        

        # optionally clamp at `clamp_exo`
        exos_before_clamp = dict() # for bookkeeping
        for i_exo, exo_val in clamp_exo.items():
            exos_before_clamp[i_exo] = exos[..., i_exo].copy()
            exos[..., i_exo] = exo_val

        # sample endogenous vars
        endos = np.zeros((1, self.n_vars,) if n is None else (n, self.n_vars), dtype=np.int64)
        endos_before_clamp = dict() # for bookkeeping
        for i_endo in range(self.n_vars):
            # use exo to seed randomness
            # compute its value
            if len(self.parents[i_endo]) == 0:
                val = categorical(self.cpts[i_endo][None, ...], exos[..., i_endo][..., None])
            else:
                val = categorical(self.cpts[i_endo][tuple(endos[..., pa] for pa in self.parents[i_endo])], exos[..., i_endo][..., None])
            if i_endo in clamp_endo:
                endos_before_clamp[i_endo] = val
                endos[..., i_endo] = clamp_endo[i_endo]
            else:
                endos[..., i_endo] = val
        if n is None or n == 1:
            endos = endos.squeeze(0)
            exos = exos.squeeze(0)
            endos_before_clamp = squeeze_dict(endos_before_clamp, 0)
            exos_before_clamp = squeeze_dict(exos_before_clamp, 0)
            
        return Sample(endos=endos,
                      exos=exos,
                      clamp_endo=clamp_endo,
                      clamp_exo=clamp_exo,
                      endos_before_clamp=endos_before_clamp,
                      exos_before_clamp=exos_before_clamp)

def nx_graph(n_vars, parents):
    graph = nx.DiGraph()
    graph.add_nodes_from(list(range(n_vars)))
    for i, pas in enumerate(parents):
        graph.add_edges_from([(pa, i) for pa in pas])
    return graph

@dataclasses.dataclass
class DAG:

    n_vars: int
    parents: int

    @property
    def nx_graph(self):
        return nx_graph(self.n_vars, self.parents)

def sample_dag(n_vars):
    # graph structure
    n_indep = np.random.randint(1, n_vars)                           # sample num of independent vars
    parents = [[] for _ in range(n_vars)]                            # data structure for parents (factors in SEM)
    for j in range(n_indep, n_vars):                                 # attach later variables sequentially
        n_parent_j = min(j, np.random.randint(1,3))                  # 1 or 2 parents
        parents[j] = np.random.choice(j, size=n_parent_j,            # sample the parents without replacement
                                      replace=False).tolist()
    return DAG(n_vars, parents)

def sample_discrete_dag(n_vars: int = 5, n_vals: int | list[int] = 2, seed: int | None = None, dag=None):
    """
    Sample a SEM with `n_vars` variables, each with `n_vals{i}` values, seeded with `seed`.

    This SEM's exogenous variables ~ U[0,1) and are used as seed to random.choice sampling from CPT.
        e.g. for graph v1 -> v2 <- v3
            v1 ~ CPT(v1)
            v3 ~ CPT(v3)
            v2 ~ CPT(v2 | v1, v3)

    Procedure:
        1. sample the number of independent vars (nodes with indegree 0)
            m: int ~ Uniform(1, n_vars)
        2. make them CPT, and sample from dirichlet.
        3. for the remaining vars
            3.1 randomly sample a number of parents between 1 and 2
            3.2 sample CPTs for that var
    """
    if seed is not None:
        np.random.seed(seed)

    # nodes
    if isinstance(n_vals, int):
        n_vals = [n_vals] * n_vars

    if dag is None:
        # if provided don't sample
        dag = sample_dag(n_vars)
    parents = dag.parents
    n_indep = sum(len(pa) == 0 for pa in parents)

    # parametric functions defining factors
    cpts = [None for _ in range(n_vars)]                             # conditional probability tables
    for j in range(n_vars):
        n_val_j = n_vals[j]
        pa_j = parents[j]
        pa_n_val = [n_vals[pa] for pa in pa_j]
        cpts[j] = np.random.dirichlet(alpha=[1] * n_val_j, 
                                    size=(reduce(int.__mul__, pa_n_val, 1),)
                                ).reshape(*(*pa_n_val, n_val_j))     # sample CPT from iid uniform dirichlet prior

    return DiscreteCPT(
        n_vars=n_vars,
        n_vals=n_vals,
        n_indep=n_indep,
        parents=parents,
        cpts=cpts
    )

if __name__ == "__main__":
    def demo():
        sem = sample_discrete_dag(seed=3, n_vals=12)
        print(sem)
        print("----")
        print("draw once")
        draw = sem.draw()
        print(draw, end="\n----\n")

        print("draw once, using same exogenous variables")
        draw_again = sem.draw(clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_again, end="\n----\n")

        print("draw once, using same exogenous variables, and intervening with do(v0)=1")
        draw_intervene = sem.draw(clamp_endo={0:1}, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_intervene, end="\n----\n")

        print("----\ntrying batch interventions")
        print("draw ten times")
        draw = sem.draw(n=10)
        print(draw, end="\n----\n")

        print("draw ten times, using same exogenous variables")
        draw_again = sem.draw(n=10, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_again, end="\n----\n")

        print("draw ten times, using same exogenous variables, and intervening with do(v0) = [0,1,...,9]")
        draw_intervene = sem.draw(n=10, clamp_endo={0:np.arange(10)}, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_intervene, end="\n----\n")

    demo()