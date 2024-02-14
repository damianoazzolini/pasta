import argparse
import sys

import cx_ax


def compute_probability(
        lb1: float, ub1: float,
        lb2: float, ub2: float,
        n_prob_facts: int, prob: float, or_and : int, pedantic: bool) -> 'tuple[float,float]':
    '''
    a) lb1 > 0 and ub1 = 1, lb2 > 0 and ub2 = 1; 
    b) lb1 > 0 and ub1 = 1, lb2 = 0 and ub2 < 1; 
    c) lb1 = 0 and ub1 < 1, lb2 > 0 and ub2 = 1;
    d) lb1 = 0 and ub1 < 1, lb2 = 0 and ub2 < 1. 
    '''
    if (lb2 > 0 and ub2 < 1) or (lb1 > 0 and ub1 < 1):
        print("No credal semantics for this configuration.")
        sys.exit()

    if lb1 > 0 and ub1 == 1 and lb2 > 0 and ub2 == 1:
        if or_and == 0:
            lp, up = cx_ax.compute_probability(max(lb1,lb2), ub1, n_prob_facts, prob)
        else:
            lp, up = cx_ax.compute_probability(min(lb1,lb2), ub1, n_prob_facts, prob)
            
    elif lb1 > 0 and ub1 == 1 and lb2 == 0 and ub2 < 1:
        # case b
        if or_and == 0:
            lp, up = cx_ax.compute_probability(lb1,ub1,n_prob_facts,prob)
        else:
            lp, up = cx_ax.compute_probability(lb2, ub2, n_prob_facts, prob)

    elif lb1 == 0 and ub1 < 1 and lb2 > 0 and ub2 == 1:
        # case c
        if or_and == 0:
            lp, up = cx_ax.compute_probability(lb2,ub2,n_prob_facts,prob)
        else:
            lp, up = cx_ax.compute_probability(lb1,ub1,n_prob_facts,prob)
            
    else:
        # case d
        if or_and == 0:
            lp, up = cx_ax.compute_probability(lb1, max(ub1,ub2), n_prob_facts, prob)
        else:
            lp, up = cx_ax.compute_probability(lb1, min(ub1,ub2), n_prob_facts, prob)

    return lp, up


if __name__ == "__main__":
    command_parser = argparse.ArgumentParser()
    command_parser.add_argument("lba", help="Lower bound", type=float)
    command_parser.add_argument("uba", help="Upper bound", type=float)
    command_parser.add_argument("lbb", help="Lower bound", type=float)
    command_parser.add_argument("ubb", help="Upper bound", type=float)
    command_parser.add_argument(
        "n", help="Number of probabilistic facts", type=int)
    command_parser.add_argument("prob", help="Probaiblity", type=float)
    command_parser.add_argument("t", help="query or (0) and (1)", type=int)
    command_parser.add_argument(
        "--pedantic", help="Pedantic mode", action="store_true")

    args = command_parser.parse_args()
    print(compute_probability(args.lba, args.uba, args.lbb,
          args.ubb, args.n, args.prob, args.t, args.pedantic))
