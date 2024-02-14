from . import lifted
import time
import random

def benchmark_cx_ax(n_vars : int, prob : float = 0.4, lb : int = 20, step : int = 5):
    '''
    Increases the vars upto n_vars
    '''
    for i in range(5, n_vars, step):
        start_time = time.time()
        lifted.conditionals_with_one_variable(i, prob, lb)
        exec_time = time.time() - start_time
        print(f"{i}, {exec_time}")


def benchmark_cx_ax_multiple_clusters_increasing_vars(n_vars : int, n_clusters : int, lb : int = 20, step : int = 5):
    '''
    Fix n_clusters and increase the vars up to n_vars
    '''
    for i in range(5, n_vars, step):
        l : 'list[list[float]]' = []
        for _ in range(0, n_clusters):
            p = random.random()
            l.append([i,p])
        # print(l)
        start_time = time.time()
        _, _, n_comp = lifted.conditionals_with_one_variable_multiple_clusters(l, 0.2)
        exec_time = time.time() - start_time
        print(f"{i}, {exec_time}, {n_comp}")


def benchmark_cx_ax_multiple_clusters_increasing_clusters(n_vars : int, n_clusters : int, lb : int = 20, step : int = 1):
    '''
    Fix n_vars per cluster and increases the clusters up to n_clusters
    '''
    for i in range(2, n_clusters + 1, step):
        l : 'list[list[float]]' = []
        for _ in range(0, i):
            p = random.random()
            l.append([n_vars,p])
        start_time = time.time()
        print(l)
        _, _, n_comp = lifted.conditionals_with_one_variable_multiple_clusters(l, 0.2)
        exec_time = time.time() - start_time
        print(f"{i}, {exec_time}, {n_comp}")


def benchmark_cx_ax_bxy_single_pair(prob : float, n_pairs : int, lb : int = 20, step : int = 1):
    '''
    c(X) | a(X), b(X,Y) by increasing the pairs (numbers of a(X))
    '''
    for n_current_pairs in range(2, n_pairs + 1, step):
        start_time = time.time()
        lifted.cx_ax_bxy_single_pair(prob, n_current_pairs)
        exec_time = time.time() - start_time
        print(f"{n_current_pairs}, {exec_time}")


def benchmark_cx_ax_bxy_multiple_pairs(prob : float, n_pairs : int, n_bi : int, lb : int = 20, step : int = 2):
    '''
    c(X) | a(X), b(X,Y) by increasing the pairs (numbers of a(X))
    and number of b(X,_) for every a(X)
    '''
    n_pairs = n_pairs * 2
    for n_current_pairs in range(2, n_pairs + 1, step):
        l : 'list[int]' = []
        for j in range(0, n_current_pairs):
            if j < n_current_pairs/2:
                l.append(1)
            else:
                l.append(n_bi)
        # print(l)
        start_time = time.time()
        _, _, n_unique_worlds, n_worlds = lifted.cx_ax_bxy_multiple_pairs(prob, l)
        exec_time = time.time() - start_time
        print(f"{n_current_pairs}, {exec_time}, {n_unique_worlds}, {n_worlds}")



if __name__ == "__main__":
    # benchmark_cx_ax(12)
    # benchmark_cx_ax_multiple_clusters_increasing_vars(11,3)
    # benchmark_cx_ax_multiple_clusters_increasing_clusters(5,5)
    # benchmark_cx_ax_bxy_single_pair(0.4,20)
    benchmark_cx_ax_bxy_multiple_pairs(0.4,3,4)