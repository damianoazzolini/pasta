import time
import sys

import importlib.util

sys.path.append("../../pasta/lifted/")

spec = importlib.util.spec_from_file_location(
    "lifted", "../../pasta/lifted/lifted.py")
past = importlib.util.module_from_spec(spec) # type: ignore
spec.loader.exec_module(past)  # type: ignore



def benchmark_cx_ax_bxy_single_pair(n_pairs: int, prob: float, lb: int = 20):
    '''
    c(X) | a(X), b(X,Y) by increasing the pairs (numbers of a(X))
    '''
    # for n_current_pairs in range(2, n_pairs + 1, step):
    start_time = time.time()
    past.cx_ax_bxy_single_pair(prob, n_pairs, lb)
    exec_time = time.time() - start_time
    print(f"{n_pairs}, {exec_time}")


if __name__ == "__main__":
    n_pairs = 5
    prob = 0.4
    lb = 20

    if len(sys.argv) > 1:
        n_pairs = int(sys.argv[1])
    if len(sys.argv) > 2:
        prob = float(sys.argv[2])
    if len(sys.argv) > 3:
        lb = int(sys.argv[3])

    benchmark_cx_ax_bxy_single_pair(n_pairs, prob, lb)
