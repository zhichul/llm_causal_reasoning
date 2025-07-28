import numpy as np
from ..graph import sample_discrete_dag

def test_seed():
    sem1 = sample_discrete_dag(n_vals=100, n_vars=3, seed=18)
    sem2 = sample_discrete_dag(n_vals=100, n_vars=3, seed=18)
    assert sem1 == sem2

    np.random.seed(12)
    sample1 = sem1.draw()
    np.random.seed(12)
    sample2 = sem2.draw()
    assert sample1 == sample2

    np.random.seed(23)
    sample1 = sem1.draw(10)
    np.random.seed(23)
    sample2 = sem2.draw(10)
    assert sample1 == sample2

def test_sample():
    n_vals = 3
    n_vars = 1000
    N = 10000
    sem = sample_discrete_dag(n_vals=n_vals, n_vars=n_vars, seed=18)
    tol = 4 # how many standard deviations before asserting error, about 6 in 100_000 chance for 4
    np.random.seed(12)
    sample = sem.draw(N)

    for var in np.random.randint(0, n_vars, 1000):
        parents = sem.parents[var]
        if len(parents) == 0:
            vals = sample.endos[..., var].reshape(-1)[None,:] # 1, samples
            possible_vals = np.array(range(n_vals))[:, None] # n_vals, 1
            counts = (vals == possible_vals).sum(-1) 
            freq = counts / N
            atol = np.sqrt(sem.cpts[var] * (1-sem.cpts[var]) / N) * tol # warning when tol standard deviations away
            assert ((freq - sem.cpts[var]) < atol).all(), (freq, sem.cpts[var], atol)
        else:
            parent_vals = [np.random.choice(n_vals) for pa in parents]
            vals = sample.endos
            for pa_val, pa in zip(parent_vals, parents):
                vals = vals[vals[:, pa] == pa_val]
            n = len(vals)
            vals = vals[:, var].reshape(-1)[None,:] # 1, samples
            possible_vals = np.array(range(n_vals))[:, None] # n_vals, 1
            counts = (vals == possible_vals).sum(-1) 
            freq = counts / N
            atol = np.sqrt(sem.cpts[var][*parent_vals] * (1-sem.cpts[var][*parent_vals]) / n) * tol # warning when tol standard deviations away
            assert ((freq - sem.cpts[var][*parent_vals]) < atol).all(), (freq, sem.cpts[var], atol)

def test_intervene_formula():
    pass

if __name__ == "__main__":
    test_seed()
    test_sample()
    test_intervene_formula()
