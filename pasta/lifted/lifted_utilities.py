import math 
import sys

def check_arguments_consistency(prob: float, lower: int, upper: int, n_vars_clusters: 'list[int]' = []) -> None:
    '''
    Check the consistency of the arguments
    '''
    stop : bool = False
    
    if lower > 0 and upper < 100:
        print("LB and UB cannot be at the same time rispectively != 0 and != 100")
        stop = True
    if lower < 0 or lower > 100:
        print("LB must be in the range [0,100]")
        stop = True
    if upper < 0 or upper > 100:
        print("UB must be in the range [0,100]")
        stop = True
    if lower > upper:
        print("LB must be less than UB")
        stop = True
    
    if isinstance(lower, float) or isinstance(upper, float):
        stop = True
        print("LB and UB must be integers")

    if prob < 0 or prob > 1:
        stop = True
        print("Probability must be in the range [0,1]")
    
    if len(n_vars_clusters) > 0:
        # all the a(i) are 1
        half = int(len(n_vars_clusters)/2)
        ais = n_vars_clusters[:half]
        # bis = n_vars_clusters[half:]
        
        for a in ais:
            if a != 1:
                print("Found a[i] != 0")
                stop = True    

    if stop:
        sys.exit()
    

def weak_compositions(boxes: int, balls: int, parent: 'tuple[int,...]' = tuple()):
    '''
    Generate weak compositions.
    From
    https://stackoverflow.com/questions/4647120/next-composition-of-n-into-k-parts-does-anyone-have-a-working-algorithm
    '''
    if boxes > 1:
        for i in range(balls + 1):
            for x in weak_compositions(boxes - 1, i, parent + (balls - i,)):
                yield x
    else:
        yield parent + (balls,)


def is_admissible(e: 'tuple[int]', constr: 'list[int]'):
    '''
    Checks that every int is less than the imposed constraint
    on cardinality. Used in generate_admissible_weak_compositions
    '''
    for el, n in zip(list(e), constr):
        if el > n:
            return False
    return True


def paired_query_constraint(world: str) -> bool:
    '''
    Supposing that the query is c(1), returns true if a(1)
    and at least one b(1,_) is true. These two elements are
    at the positions 0 and len(world) / 2
    '''
    return int(world[0]) > 0 and int(world[int(len(world)/2)]) > 0


def generate_admissible_weak_compositions(max_vars_per_clusters: 'list[int]', n_tot_vars: int, pair_constraint: bool = False):
    '''
    Generate admissible weak compositions where admissibility is
    defined by max_vars_per_cluster
    '''
    r: 'list[tuple[int,...]]' = []
    for el in weak_compositions(len(max_vars_per_clusters), n_tot_vars):
        if not pair_constraint:
            if is_admissible(el, max_vars_per_clusters):
                r.append(el)
        else:
            if is_admissible(el, max_vars_per_clusters) and paired_query_constraint(el):
                r.append(el)
    return r


def count_variables(bin_s: str) -> int:
    '''
    Counts the number of 1 in a binary string
    '''
    return bin_s.count('1')


def count_overlaps(a: str, b: str) -> int:
    '''
    Given two binary strings a and b, counts the 
    number of matching 1
    '''
    o = 0
    for i in range(0, len(a)):
        if int(a[i]) * int(b[i]) == 1:
            o += 1
    return o


def number_of_comb_overlaps(n_vars: int, k: int, i: int) -> 'tuple[int,list[str]]':
    '''
    Computes the number of combinations such that the number 
    of 1s in correspondent positions in a string of length n_pairs/2 
    of the representation of a world (we call this value overlap)
    equals k and there are i 1s in the string. Naive enumeration
    but currently i do not have a better method
    '''
    upper_b = 2**n_vars  # since 2^n -1 = 11...11
    count = 0
    lv: 'list[str]' = []
    for id in range(0, upper_b):
        s = format(id, f'0{n_vars}b')
        a = s[0:int(len(s)/2)]
        b = s[int(len(s)/2):]
        # print(f"{s}, i = {count_variables(s)}, k = {count_overlaps(a,b)}")
        if count_variables(s) == i and count_overlaps(a, b) == k:
            count += 1
            lv.append(s)
    return count, lv


def number_of_comb_overlaps_cxy_ax_bxy(n_vars: int, n_pairs: int, i: int) -> 'tuple[int,list[str]]':
    '''
    Computes the number of combinations such that the number 
    of 1s in correspondent positions in a string of length n_vars * 2
    of the representation of a world (we call this value overlap)
    equals k and there are i 1s in the string. Naive enumeration
    but currently i do not have a better method
    '''
    n_vars = n_vars * 2
    upper_b = 2**n_vars  # since 2^n -1 = 11...11
    count = 0
    lv: 'list[str]' = []
    for id in range(0, upper_b):
        s = format(id, f'0{n_vars}b')
        if int(s[0]) == 1 and int(s[int(len(s)/2)]) == 1: # a(1) is true
            a = s[0:int(len(s)/2)]
            b = s[int(len(s)/2):]
            # print(f"{s}, i = {count_variables(s)}, k = {count_overlaps(a,b)}")
            if count_variables(s) == i and count_overlaps(a, b) == n_pairs:
                count += 1
                lv.append(s)
    return count, lv


def count_pairs_wc(world: 'tuple[int]') -> int:
    '''
    Counts the pairs of a(i) b(i,_) where both are > 0
    '''
    c = 0
    l2 = int(len(world)/2)
    for i in range(0, l2):
        c += int((int(world[i]) * int(world[i + l2])) > 0)
    return c


def count_b1_wc(world: 'tuple[int]') -> int:
    '''
    Counts the b(1,_) in the world 
    '''
    l2 = int(len(world)/2)
    return int(world[l2])


def compute_binom(world: 'tuple[int]', n_vars_cluster: 'list[int]') -> int:
    '''
    Computes the number of worlds with the same structure
    that can be created, i.e., the product of the binomial
    coefficients (n k) where n is n_vars_cluster[i] while 
    k is world[i]
    '''
    prod = 1
    for i in range(0, len(world)):
        prod = prod * math.comb(n_vars_cluster[i], int(world[i]))
    return prod


def compute_binom_no_b1(world: 'tuple[int]', n_vars_cluster: 'list[int]') -> int:
    '''
    Computes the number of worlds with the same structure
    that can be created, i.e., the product of the binomial
    coefficients (n k) where n is n_vars_cluster[i] while 
    k is world[i]. Does not consider the values on b(1,1),
    since it is always 1
    '''
    prod = 1
    world_l = list(world)
    # print(world_l, n_vars_cluster)
    n_vars_cluster_copy = n_vars_cluster.copy()
    pos_b11 = int(len(n_vars_cluster)/2)
    if world_l[pos_b11] == 1 and n_vars_cluster_copy[pos_b11] > 1:
        n_vars_cluster_copy[pos_b11] = 1
    elif n_vars_cluster_copy[pos_b11] > 1 and (world_l[pos_b11] != n_vars_cluster_copy[pos_b11]):
        n_vars_cluster_copy[pos_b11] -= 1
        world_l[pos_b11] -= 1
    # if n_vars_cluster_copy[pos_b11] == 1:
    #     world[pos_b11] = 1

    # print(n_vars_cluster_copy, world)
    for i in range(0, len(world_l)):
        prod = prod * math.comb(n_vars_cluster_copy[i], int(world_l[i]))
    return prod