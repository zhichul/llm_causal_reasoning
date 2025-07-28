from collections import defaultdict
import numpy as np

from causal_graph.discrete_cpt import sample_discrete_dag

import logging
logger = logging.getLogger()

def variable_elimination(cpts: list[np.ndarray], parents: list[list[int]], order: list[int], renormalize=False):
    """
    Assumes that parents is in sorted order lowest to highest. Parents includes the node itself as well, as the last element.
    """
    parents = {i: {pa: None for pa in pas} for i, pas in enumerate(parents)}
    cpts = {i: cpt for i, cpt in enumerate(cpts)}
    next_id = len(parents)
    constants = 1
    logger.debug(f"#factors={len(cpts)}")
    for var in order:
        operands = []
        in_indices = []
        out_indices = dict()
        eliminated = []
        for i in cpts:
            pas = parents[i]
            cpt = cpts[i]
            if var in pas:
                eliminated.append(i)
                operands.append(cpt)
                in_indices.append("".join([chr(97+pa) for pa in pas]))
                for pa in pas:
                    if pa != var:
                        out_indices[chr(97+pa)] = pa
        subscript = f'{",".join(in_indices)}->{"".join(list(out_indices.keys()))}'
        logger.debug(subscript)
        new_factor = np.einsum(subscript, *operands)
        logger.debug([operand.shape for operand in operands] + [new_factor.shape])

        new_pas = list(out_indices.values())

        logger.debug(f"-{len(eliminated)} factors of sizes {[len(parents[i]) for i in eliminated]}")
        for i in eliminated:
            del parents[i]
            del cpts[i]

        if len(new_pas) == 0:
            # summed out the entire factor into a scaler
            constants = constants * new_factor
            logger.debug(f"produced a constant")
        else:
            cpts[next_id] = new_factor
            parents[next_id] = {pa: None for pa in new_pas}
            next_id += 1
            logger.debug(f"+1 factor of size [{len(new_pas)}]")
        logger.debug(f"#factors={len(cpts)}")

    dims = [len(cpt.shape) for cpt in cpts.values()]
    assert len(set(dims)) == 1
    shape = [cpt.shape for cpt in cpts.values()]
    assert len(set(shape)) == 1
    if len(cpts) > 1:
        ret = 1.
        for cpt in cpts.values():
            ret = ret * cpt
        result = [ret, list(next(iter(parents.values())).keys())]
    else:
        result = [next(iter(cpts.values())) * constants, list(next(iter(parents.values())).keys())]
    if renormalize:
        result[0] = result[0] / result[0].sum()
    return result

if __name__ == "__main__":
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S')    
    def demo():
        sem = sample_discrete_dag(seed=3, n_vals=12)
        print(sem)
        marginal, parents = variable_elimination([sem.cpts[0], sem.cpts[3]], [sem.iparents[0], sem.iparents[3]], [0])
        manual_marginal = np.matmul(sem.cpts[0], sem.cpts[3])
        np.testing.assert_allclose(marginal, manual_marginal)
        print("test 1 done, leaf to root, two nodes")
        marginal, parents = variable_elimination([sem.cpts[0], sem.cpts[3], sem.cpts[2], sem.cpts[4]], [sem.iparents[0], sem.iparents[3], sem.iparents[2], sem.iparents[4]], [4, 2, 0])
        np.testing.assert_allclose(marginal, manual_marginal)
        print("test 2 done, added extra distractor nodes that don't matter")

        # this is a graph with two diamonds
        sem = sample_discrete_dag(seed=15, n_vals=12)
        print(sem)
        marginal3, parents3 = variable_elimination([sem.cpts[0], sem.cpts[1], sem.cpts[2], sem.cpts[3]], [sem.iparents[0], sem.iparents[1], sem.iparents[2], sem.iparents[3]], [0, 1, 2])
        manual_joint_12 = (sem.cpts[0][:, None, None] * sem.cpts[1][:,:,None] * sem.cpts[2][:, None, :]).sum(0)
        manual_marginal_3 = np.matmul(manual_joint_12.reshape(-1), sem.cpts[3].reshape(-1, sem.cpts[3].shape[-1]))
        np.testing.assert_allclose(marginal3, manual_marginal_3)
        print("test 3 done, diamond")
        marginal3, parents3 = variable_elimination([sem.cpts[0], sem.cpts[1], sem.cpts[2], sem.cpts[3]], [sem.iparents[0], sem.iparents[1], sem.iparents[2], sem.iparents[3]], [2, 1, 0])
        np.testing.assert_allclose(marginal3, manual_marginal_3)
        print("test 4 done, diamond but change order")


    demo()