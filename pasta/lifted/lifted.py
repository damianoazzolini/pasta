import math

import lifted_utilities
import lifted_deltas


def cx_ax_one_probability_cluster(n_vars : int, prob : float = 0.5, lb : int = 60, ub : int = 100) -> 'tuple[float,float]':
    '''
    Conditionals of the form:
    p::a(l..u).
    (c(X) | a(X))[lb,ub].
    Supposing that all the worlds have at least one stable model.
    '''
    l = 1
    u = n_vars
    lp : float = 0
    up: float = 0
    n = u - l + 1
    query = "c(1)"
    for i in range(0,n):
        # print(f'it {i}')
        dl = lifted_deltas.delta_ax(0, i + 1, lb, ub, query)
        du = lifted_deltas.delta_ax(1, i + 1, lb, ub, query)
        # print(dl,du)
        lp += math.comb(n-1,i) * dl * prob**(i+1) * (1-prob)**(n-i-1)
        up += math.comb(n-1, i) * du * prob**(i+1) * (1-prob)**(n-i-1)
    
    return lp, up


def cx_ax_multiple_probability_clusters(clusters : 'list[list[float]]', p_fix : float):
    '''
    Conditionals of the form:
    p::a(l..u).
    (c(X) | a(X))[lb,ub].
    where the a(X) are grouped in different clusters of probabilities.
    '''
    l : int = 1
    n_vars_cluster : 'list[int]' = [int(item[0]) for item in clusters]
    u : int = sum(n_vars_cluster)
    n : int = u - l + 1
    # print(n_vars_cluster)
    total_probability_l : float = 0
    total_probability_u : float = 0
    n_compositions_generated : int = 0
    lb : int = 20
    ub : int = 20
    for i in range(0, n + 1):
        dl = lifted_deltas.delta_ax(0, i + 1, lb, ub, "c(1)")
        du = lifted_deltas.delta_ax(1, i + 1, lb, ub, "c(1)")
        
        lwc = lifted_utilities.generate_admissible_weak_compositions(n_vars_cluster, i)
        
        n_compositions_generated = n_compositions_generated + len(lwc)
        
        # print(f"--- {i} ---")
        # print(dl, du, lwc)
        for wc in lwc:
            # print(wc)
            prod = 1
            for index, selected_current_cluster in enumerate(list(wc)):
                n_tot_current_cluster : int = int(clusters[index][0])
                prob_current_cluster : float = clusters[index][1]
                # print(f"comb: {math.comb(n_tot_current_cluster, selected_current_cluster)}")
                contr = (prob_current_cluster**selected_current_cluster) * \
                    ((1 - prob_current_cluster)**(n_tot_current_cluster - selected_current_cluster))
                # print(f"contr: {contr}")
                prod = prod * \
                    math.comb(n_tot_current_cluster, selected_current_cluster) * \
                    contr
            # print(f"probability: {prod}")
            # print(f"total contr: {prod * p_fix}")
            total_probability_l = total_probability_l + prod * p_fix * dl
            total_probability_u = total_probability_u + prod * p_fix * du

    # print(total_probability_l, total_probability_u, n_compositions_generated)
    return total_probability_l, total_probability_u, n_compositions_generated


def cxy_ax_bxy_multiple_bi(prob: float, n_vars_clusters: 'list[int]', lower: int = 0, upper: int = 100) -> 'tuple[float,float,int,int]':
    '''
    Conditionals (c(X,Y) | a(X), b(X,Y))[lb,ub]
    n_vars_clusters is a list of int, where each element at position i 
    represents the number of (i) ... b(i,_).
    By default, the query is c(1,1).
    Example: n_vars_clusters = [1,1,2,2] = 
    a(1), a(2), b(1,1), b(1,2), b(2,1), b(2,2)
    '''
    # if n_vars_clusters[0] == 0:
    #     print("n_vars_b_i[0] must be at least 1")
    #     sys.exit()
    
    lifted_utilities.check_arguments_consistency(prob, lower, upper, n_vars_clusters)
    
    # TODO: praticamente uguale a cx_ax_bxy_multiple_pairs
    up: float = 0
    lp: float = 0
    tot_vars = sum(n_vars_clusters)
    n_worlds = 0
    n_unique_worlds = 0 # i.e., number of ASP calls
    pairs = int(len(n_vars_clusters) / 2)
    # n_vars_clusters_without_b_11 = n_vars_clusters.copy()
    # n_vars_clusters_without_b_11[int(len(n_vars_clusters)/2)] -= 1
    # print(n_vars_clusters)

    for i in range(2, tot_vars + 1):
        lwc = lifted_utilities.generate_admissible_weak_compositions(n_vars_clusters, i, True)
        # print(f"{i}: {lwc} ({len(lwc)})")

        for n_pairs in range(1, pairs + 1):
            for n_b1 in range(1, n_vars_clusters[int(len(n_vars_clusters)/2)] + 1):
                for wc in lwc:
                    # mul = 1
                    if (lifted_utilities.count_pairs_wc(wc) == n_pairs) and (lifted_utilities.count_b1_wc(wc) == n_b1):

                        b = lifted_utilities.compute_binom_no_b1(wc, n_vars_clusters)
                        # print(f"n_pairs: {n_pairs}, n_b1: {n_b1}, wc: {wc}, b: {b}")
                        # b = lifted_utilities.compute_binom_no_b1(wc, n_vars_clusters)
                        # b = 1
                        n_unique_worlds += 1
                        n_worlds += b
                        # print(b)
                        # mul = mul * b
                        dl = 0
                        du = lifted_deltas.delta_cxy_axbxy_k(1, wc, lower, upper, 'c(1,1)')
                        world_probability = (prob**i) * ((1 - prob)**(tot_vars - i))
                        # print(world_probability, b)
                        if du != 0:
                            up += world_probability * b
                            dl = lifted_deltas.delta_cxy_axbxy_k(
                                0, wc, lower, upper, 'c(1,1)')
                            lp += world_probability * b * dl
                        # print(du, dl, b, world_probability)

    return lp, up, n_unique_worlds, n_worlds


def cx_ax_bxy_single_pair(prob: float, n_vars: int, lower: int = 60, upper: int = 100, formula: bool = True) -> 'tuple[float,float]':
    '''
    Conditionals c(X) | a(X), b(X,Y) with all the facts
    with the same probability and 1 b(X,Y) for every a(X)
    and viceversa
    upperprob(q) = p_k sum_{i=0}^{n} sum_{k = 0} ^ {n/2} overline{delta}_{ik}(q) cdot rho(n,k,i) cdot p^i cdot (1-p)^{n-i}
    '''
    if n_vars % 2 != 0:
        print("The number of variables must be even")
        # import sys
        # sys.exit()
    
    n_vars = n_vars - 2  # remove the c(i) -> a(i) b(i,_)
    pk = prob * prob
    up = 0
    lp = 0
    for i in range(0, n_vars + 1):
        for k in range(0, int(n_vars / 2) + 1):
            # print(f"n_vars: {n_vars}, k: {k}, i: {i}")
            if formula:
                rho, lw = lifted_utilities.number_of_comb_overlaps_formula(n_vars, k, i)
                # print(f"rho, lw: {rho}, {lw}")
            else:
                rho, lw = lifted_utilities.number_of_comb_overlaps(n_vars, k, i)
                # print(f"rho, lw: {rho}, {lw}")
                lw = lw[0] if len(lw) > 0 else ''
            # print(rho, lw)

            # if len(lw) > 0:
            if rho > 0:
                du = lifted_deltas.delta_cx_axbxy(1, lw, lower, upper, "c(1)")
                if du != 0:
                    up += rho * (prob**i) * ((1 - prob)**(n_vars - i)) * du
                    lp += rho * (prob**i) * ((1 - prob)**(n_vars - i)) * \
                        lifted_deltas.delta_cx_axbxy(0, lw, lower, upper, "c(1)")
                        # lifted_deltas.delta_cx_axbxy(
                            # 0, lw, lower, upper, "c(1)")
                        
    
    return lp * pk, up * pk


def cx_ax_bxy_multiple_pairs(prob: float, n_vars_cluster: 'list[int]', lower: int = 60, upper: int = 100) -> 'tuple[float,float,int,int]':
    '''
    Conditionals (c(X) | a(X), b(X,Y)) with all the facts
    with the same probability and more than 1 b(X,Y)
    for every a(X)
    '''
    # n_pairs = n_pairs - 2  # remove the c(i)
    # pk = prob * prob
    up : float = 0
    lp : float = 0
    tot_vars = sum(n_vars_cluster)
    n_worlds = 0
    n_unique_worlds = 0 # i.e., number of ASP calls
    na = int(len(n_vars_cluster) / 2)
    
    for i in range(2, tot_vars + 1):
        lwc = lifted_utilities.generate_admissible_weak_compositions(n_vars_cluster, i, True)
        # print(f"{i}: {lwc} ({len(lwc)})")
        # t_up = 0
        # t_lp = 0
        # mul = 1
        # for wc in lwc:
        #     print(wc, count_pairs_wc(wc), count_b1_wc(wc))
        for n_pairs in range(1, na + 1):
            for n_b1 in range(1, tot_vars + 1):
                for wc in lwc:
                    # mul = 1
                    if (lifted_utilities.count_pairs_wc(wc) == n_pairs) and (lifted_utilities.count_b1_wc(wc) == n_b1):
                        # print(f"n_pairs: {n_pairs}, n_b1: {n_b1}, wc: {wc}")
                        b = lifted_utilities.compute_binom(wc, n_vars_cluster)
                        n_unique_worlds += 1
                        n_worlds += b
                        # print(b)
                        # mul = mul * b
                        du = lifted_deltas.delta_cx_axbxy_k(1, wc, lower, upper, 'c(1)')
                        if du != 0:
                            world_probability = (prob**i) * ((1 - prob)**(tot_vars - i))
                            # print(world_probability, b)
                            up += world_probability * b
                            dl = lifted_deltas.delta_cx_axbxy_k(
                                0, wc, lower, upper, 'c(1)')
                            lp += world_probability * b * dl

    return lp, up, n_unique_worlds, n_worlds


# def almost_equal(comp_l : float, comp_u : float, real_l : float, real_u : float) -> bool:
#     return (math.isclose(comp_l, real_l, abs_tol = 1e-6)) and (math.isclose(comp_u, real_u, abs_tol = 1e-7))


# def test_cx_ax_bxy_single_pair():
#     l, u = cx_ax_bxy_single_pair(0.4,6)
#     expected_l, expected_u = 0.1559040, 0.16 
#     assert True == almost_equal(
#         l, u, expected_l, expected_u), f"{l, u} != {expected_l, expected_u}"
    
#     l, u = cx_ax_bxy_single_pair(0.3,6)
#     expected_l, expected_u = 0.0892709, 0.089 
#     assert True == almost_equal(l, u, expected_l, expected_u), f"{l, u} != {expected_l, expected_u}"
    
#     l, u = cx_ax_bxy_single_pair(0.3, 6, lower=20)
#     expected_l, expected_u = 0.0745289, 0.09
#     assert True == almost_equal(
#         l, u, expected_l, expected_u), f"{l, u} != {expected_l, expected_u}"


if __name__ == "__main__":
    # conditionals_with_one_variable_multiple_clusters([[2,0.2],[2,0.3]], 0.2)
    # print(conditionals_with_one_variable(100))
    # stress_test_conditionals_with_one_variable()
    # print(conditionals_with_one_variable(2,0.16)) 
    # print(cxy_ax_by(6,0.4)) 
    # c = 0
    # for i in range(1,7):
    #     print(f"-- {i} --")
    #     wc = generate_admissible_weak_compositions([1,1,2,2],i)
    #     print(*wc,sep='\n')
    #     c += len(wc)
    # print(c)
    # print(number_of_comb_overlaps(10, 1, 6))
    # test_cx_ax_bxy_single_pair()
    print(cx_ax_bxy_single_pair(0.3, 6, lower=40))
    print(cx_ax_bxy_single_pair(0.3, 6, lower=40, formula=False))
    # print(cx_ax_bxy_multiple_pairs(0.4, [1,3], lower=40))
    # print(cxy_ax_bxy_multiple_bi(0.4,[1,1,2,2],lower=40))
    # print(cxy_ax_bxy_multiple_bi(0.4,[1,1,1,1,1, 3,2,2,1,1],lower=0, upper=80))