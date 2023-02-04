import time
import random
import sys

import importlib.util

sys.path.append("../../pasta/lifted/")

spec = importlib.util.spec_from_file_location(
    "lifted", "../../pasta/lifted/lifted.py")
past = importlib.util.module_from_spec(spec) # type: ignore
spec.loader.exec_module(past) # type: ignore


def benchmark_cx_ax_multiple_clusters_increasing_vars(n_vars: int, n_clusters: int):
    '''
    Fix n_clusters and increase the vars up to n_vars
    '''
    # for i in range(5, n_vars, step):
    l: 'list[list[float]]' = []
    for _ in range(0, n_clusters):
        p = random.random()
        l.append([n_vars, p])
    # print(l)
    start_time = time.time()
    _, _, n_comp = past.cx_ax_multiple_probability_clusters(l, 0.2)
    exec_time = time.time() - start_time
    print(f"{n_vars}, {exec_time}, {n_comp}")


if __name__ == "__main__":
    n_vars = 5
    n_clusters = 4
    lb = 20
    # step = 5

    if len(sys.argv) > 1:
        n_vars = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_clusters = int(sys.argv[2])

    benchmark_cx_ax_multiple_clusters_increasing_vars(n_vars, n_clusters)
