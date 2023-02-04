import time
import sys

import importlib.util

sys.path.append("../../pasta/lifted/")

spec = importlib.util.spec_from_file_location(
    "lifted", "../../pasta/lifted/lifted.py")
past = importlib.util.module_from_spec(spec) # type: ignore
spec.loader.exec_module(past) # type: ignore


def benchmark_cx_ax_bxy_multiple_pairs(prob: float, n_pairs: int, n_bi: int, lb: int = 20, step: int = 2):
    '''
    c(X) | a(X), b(X,Y) by increasing the pairs (numbers of a(X))
    and number of b(X,_) for every a(X)
    '''
    n_pairs = n_pairs * 2
    # for n_current_pairs in range(2, n_pairs + 1, step):
    l: 'list[int]' = []
    for j in range(0, n_pairs):
        if j < n_pairs/2:
            l.append(1)
        else:
            l.append(n_bi)
    # print(l)
    start_time = time.time()
    _, _, n_unique_worlds, n_worlds = past.cx_ax_bxy_multiple_pairs(prob, l)
    exec_time = time.time() - start_time
    print(f"{n_pairs}, {n_bi}, {exec_time}, {n_unique_worlds}, {n_worlds}")


if __name__ == "__main__":
    n_pairs = 2
    n_bi = 2
    prob = 0.4
    lb = 20
    step = 1

    if len(sys.argv) > 1:
        n_pairs = int(sys.argv[1])
    if len(sys.argv) > 2:
        n_bi = int(sys.argv[2])
    # if len(sys.argv) > 3:
    #     lb = int(sys.argv[3])
    # if len(sys.argv) > 4:
    #     step = int(sys.argv[4])

    benchmark_cx_ax_bxy_multiple_pairs(prob, n_pairs, n_bi)
