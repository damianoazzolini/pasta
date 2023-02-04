
import time
import sys

import importlib.util

sys.path.append("../../pasta/lifted/")

spec = importlib.util.spec_from_file_location(
    "lifted", "../../pasta/lifted/lifted.py")
past = importlib.util.module_from_spec(spec) # type: ignore
spec.loader.exec_module(past) # type: ignore


def benchmark_cx_ax(n_vars: int, prob: float = 0.4, lb: int = 20, step: int = 5):
    '''
    c(X)|a(X)
    Increases the vars upto n_vars
    '''
    # for i in range(5, n_vars, step):
    start_time = time.time()
    past.cx_ax_one_probability_cluster(n_vars, prob, lb)
    exec_time = time.time() - start_time
    print(f"{n_vars}, {exec_time}")


if __name__ == "__main__":
    n_vars = 11
    lb = 20
    step = 5
    prob = 0.4

    if len(sys.argv) > 1:
        n_vars = int(sys.argv[1])
    if len(sys.argv) > 2:
        lb = int(sys.argv[2])
    if len(sys.argv) > 3:
        step = int(sys.argv[3])

    benchmark_cx_ax(n_vars, prob, lb, step)
