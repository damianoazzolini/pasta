import argparse
import math
import sys

def compute_probability(
    lb : float, ub : float, n_prob_facts : int, prob : float) -> 'tuple[float,float]':
    '''
    If lb > 0 and ub = 1: upperprob = q, lowerprob $n_a = \lceil lb \cdot n_a \rceil$
    If lb = 0 and ub < 1: upperprob ub*na \geq 1, lowerprob = 0
    '''
    if lb > 0 and ub < 1:
        print("No credal semantics for lb > 0 and ub < 1.")
        sys.exit()
    
    up = 0
    lp = 0
    if lb > 0 and ub == 1:
        up = prob
        for k in range(0, n_prob_facts):
            if k + 1 == math.ceil(lb * (k+1)):
                p = (prob**k) * ((1 - prob)**(n_prob_facts - 1 - k))
                lp += math.comb(n_prob_facts - 1, k) * p 
        lp = lp * prob
    else:
        lp = 0
        for k in range(0, n_prob_facts):
            if ub*(k + 1) >= 1:
                p = (prob**k) * ((1 - prob)**(n_prob_facts - 1 - k))
                up += math.comb(n_prob_facts - 1, k) * p 
        up = up * prob

    return lp, up


if __name__ == "__main__":
    command_parser = argparse.ArgumentParser()
    command_parser.add_argument("lb", help="Lower bound", type=float)
    command_parser.add_argument("ub", help="Upper bound", type=float)
    command_parser.add_argument("n", help="Number of probabilistic facts", type=int)
    command_parser.add_argument("prob", help="Probability", type=float)
    args = command_parser.parse_args()
    print(compute_probability(args.lb,args.ub,args.n,args.prob))
    
