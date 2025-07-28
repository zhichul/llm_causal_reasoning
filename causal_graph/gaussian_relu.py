from __future__ import annotations
import dataclasses
import numpy as np
from scipy import stats


class LeakyReLU:

    def __init__(self, rate):
        self.rate = rate
    
    def __repr__(self):
        return f'LeakyReLU({self.rate})'

    def __call__(self, arr: np.ndarray):
        return  np.maximum(self.rate * arr, arr)

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

@dataclasses.dataclass
class SimpleTransformGaussianSEM:

    n_vars: int                                         # number of variables
    n_indep: int                                        # number of independent vars             
    parents: list[list | np.ndarray]                    # parents (in canonical index order)                            (2d array)
    weights: list[list | np.ndarray]                    # (linear) coefficients of parents                              (2d array)
    variances: np.ndarray                               # variances of the exogenous vars                               (1d array)
    nonlinearity: callable[[np.ndarray], np.ndarray]    # optional nonlinearity

    def __str__(self):
        if getattr(self, '_default_varnames', None) is None:
            self._default_varnames = [f'#{i}' for i in range(self.n_vars)]
        return self.to_str(self._default_varnames)

    def to_str(self, varnames: list[str]):
        lines = []
        for i, (parents, weights, variance) in enumerate(zip(self.parents, self.weights, self.variances)):
            if len(parents) == 0:
                lines.append(f"{varnames[i]} ~ N(0, {variance})")
            else:
                lines.append(f"{varnames[i]} ~ {self.nonlinearity}([{' '.join([varnames[p] for p in parents])}] * {weights} + N(0, {variance}))")
        return "\n".join(lines)
    
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

        # sample exogenous vars
        exos = stats.norm.rvs(loc=0, scale=np.sqrt(self.variances), size=None if n is None else (n, self.n_vars))
        
        # optionally clamp at `clamp_exo`
        exos_before_clamp = dict() # for bookkeeping
        for i_exo, exo_val in clamp_exo.items():
            exos_before_clamp[i_exo] = exos[..., i_exo].copy()
            exos[..., i_exo] = exo_val

        # sample endogenous vars
        endos = np.zeros((self.n_vars,) if n is None else (n, self.n_vars))
        endos_before_clamp = dict() # for bookkeeping
        for i_endo in range(self.n_vars):
            # compute its value
            if len(self.parents[i_endo]) == 0:
                val = exos[..., i_endo]
            else:
                val = self.nonlinearity(np.matmul(endos[..., self.parents[i_endo]], self.weights[i_endo]) + exos[..., i_endo])
            if i_endo in clamp_endo:
                endos_before_clamp[i_endo] = val
                endos[..., i_endo] = clamp_endo[i_endo]
            else:
                endos[..., i_endo] = val
                
        return Sample(endos=endos,
                      exos=exos,
                      clamp_endo=clamp_endo,
                      clamp_exo=clamp_exo,
                      endos_before_clamp=endos_before_clamp,
                      exos_before_clamp=exos_before_clamp)

def sample_sem(n_vars: int = 5, seed: int | None = None):
    """
    Sample a SEM with `n_vars` variables, seeded with `seed`.

    Adapted from Lampinen et al., 2023
        Passive learning of active causal strategies in agents and language models.

    This SEM has gaussian exogenous variables, and a LeakyReLU nonlinearity:
        e.g. for graph v1 -> v2 <- v3
            v1 ~ some gaussian N1
            v3 ~ some gaussian N2
            v2 ~ LeakyReLU(w1 * v1 + w2 * v2 + some gaussian N3)

    Procedure:
        1. sample the number of independent vars (nodes with indegree 0)
            m: int ~ Uniform(1, n_vars)
        2. make them zero-centered gaussian, and sample their variances.
        3. for the remaining vars
            3.1 randomly sample a number of parents between 1 and 2
            3.2 sample weights for those parents
            3.3 sample the variance of its own Gaussian noise
    """
    if seed is not None:
        np.random.seed(seed)

    # graph structure
    n_indep = np.random.randint(1, n_vars)                           # sample num of independent vars
    parents = [[] for _ in range(n_vars)]                            # data structure for parents (factors in SEM)
    for j in range(n_indep, n_vars):                                 # attach later variables sequentially
        n_parent_j = min(j, np.random.randint(1,3))                  # 1 or 2 parents
        parents[j] = np.random.choice(j, size=n_parent_j,         # sample the parents without replacement
                                      replace=False)                 

    # parametric functions defining factors
    weights = [[] for _ in range(n_vars)]                                                  # parent weights in linear combination
    for j in range(n_indep, n_vars):
        weights[j] = stats.uniform.rvs(loc=-2, scale=4, size=len(parents[j]))        # parent weights sampled from unifrom [-2., 2.]
    variances = stats.truncnorm.rvs(a=-2, b=np.inf, loc=0.5, scale=0.25, size=(n_vars,))   # sample exogenous variable variances
    nonlinearity = LeakyReLU(rate=0.2) 

    return SimpleTransformGaussianSEM(
        n_vars=n_vars,
        n_indep=n_indep,
        parents=parents,
        weights=weights,
        variances=variances,
        nonlinearity=nonlinearity
    )

if __name__ == "__main__":
    def demo():
        sem = sample_sem(seed=3)
        print(sem)

        print("----")
        print("draw once")
        draw = sem.draw()
        print(draw, end="\n----\n")


        print("draw once, using same exogenous variables")
        draw_again = sem.draw(clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_again, end="\n----\n")

        print("draw once, using same exogenous variables, and intervening with do(v0)=5")
        draw_intervene = sem.draw(clamp_endo={0:5.}, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_intervene, end="\n----\n")

        print("----\ntrying batch interventions")
        print("draw ten times")
        draw = sem.draw(n=10)
        print(draw, end="\n----\n")

        print("draw ten times, using same exogenous variables")
        draw_again = sem.draw(n=10, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_again, end="\n----\n")

        print("draw ten times, using same exogenous variables, and intervening with do(v0) = [0,1,2,...,9]")
        draw_intervene = sem.draw(n=10, clamp_endo={0:np.arange(10)}, clamp_exo=dict(enumerate(draw.exos.T)))
        print(draw_intervene, end="\n----\n")

    demo()