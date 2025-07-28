import numpy as np
from scipy.stats import permutation_test

def indistinguishable_from_best(scores, 
                                alpha=0.05, 
                                n_resamples=9999,
                                higher_is_better=True,
                                rng=0):
    """
    Parameters
    ----------
    scores : array-like, shape (m, N)
        Row i holds the scores that *system i* received on each test case.
    alpha  : float
        Per-comparison significance level (no multiplicity adjustment).
    n_resamples : int
        Sign-flip permutations; an exact test is run automatically when 2**N
        is smaller than n_resamples.
    higher_is_better : bool
        If False, lower mean scores are considered better.
    rng : int or np.random.Generator
        Seed / Generator for reproducibility.

    Returns
    -------
    keepers : ndarray
        Indices of systems whose performance is NOT significantly different
        from the best system at the given α.
    pvals : ndarray
        Raw p-values for each system vs. the best.
    """
    scores = np.asarray(scores, dtype=float)        # shape (m, N)
    if not higher_is_better:
        scores = -scores                            # flip the direction

    means   = scores.mean(axis=1)                   # mean per *row*
    best_ix = means.argmax()
    m = scores.shape[0]
    pvals = np.ones(m)                              # p-values vs. best

    def stat(x, y):                                 # paired mean difference
        return np.mean(x - y)

    for j in range(m):
        if j == best_ix:
            continue
        res = permutation_test(
            data=(scores[best_ix], scores[j]),
            statistic=stat,
            permutation_type='samples',             # sign flips
            alternative='two-sided',
            n_resamples=n_resamples,
            random_state=rng
        )
        pvals[j] = res.pvalue
    keep = (pvals > alpha).tolist()       # fail to reject H₀
    if not higher_is_better:
        means = -means
    return keep, pvals, means