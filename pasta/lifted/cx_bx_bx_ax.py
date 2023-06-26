import argparse
import math
import sys

import clingo


def delta(mode: int,
        lb1: float, ub1: float,
        lb2 :float, ub2 : float,
        n_prob_facts: int, pedantic : bool) -> int:
    '''
    mode = 0 lower - cautious
    mode = 1 upper - brave
    '''
    if mode == 0:
        ctl = clingo.Control(["--enum-mode=cautious"])
    else:
        ctl = clingo.Control(["--enum-mode=brave"])

    # ctl.add('base', [], "a(1).")
    ctl.add('base', [], "0{ c(X)  }1 :-  b(X).")
    ctl.add('base', [], "0{ b(X)  }1 :-  a(X).")
    
    if lb1 != 0:
        s = ":- #count{X: b(X)} = H, #count{X:c(X) , b(X)} = FH, 100*FH < " + str(int(lb1*100)) + "*H."
        if pedantic:
            print(s)
        ctl.add('base', [], s)
    if lb2 != 0:
        s = ":- #count{X: a(X)} = H, #count{X:b(X) , a(X)} = FH, 100*FH < " + str(int(lb2*100)) + "*H."
        if pedantic:
            print(s)
        ctl.add('base', [], s)

    if ub1 != 1:
        s = ":- #count{X: b(X)} = H, #count{X:c(X) , b(X)} = FH, 100*FH > " + str(int(ub1*100)) + "*H."
        if pedantic:
            print(s)
        ctl.add('base', [], s)
    if ub2 != 1:
        s = ":- #count{X: a(X)} = H, #count{X:b(X) , a(X)} = FH, 100*FH > " + str(int(ub2*100)) + "*H."
        if pedantic:
            print(s)
        ctl.add('base', [], s)
        
    for i in range(1, n_prob_facts + 2):
        ctl.add('base', [], f"a({i}).")
        if pedantic:
            print(f"a({i}).")

    ctl.ground([("base", [])])

    opt_m: str = ""
    with ctl.solve(yield_=True) as handle:  # type: ignore
        for m in handle:  # type: ignore
            opt_m = str(m)  # type: ignore
            handle.get()  # type: ignore

    return 1 if "c(1)" in opt_m.split(' ') else 0


def compute_probability(
        lb1: float, ub1: float,
        lb2 :float, ub2 : float,
        n_prob_facts: int, prob: float, pedantic : bool) -> 'tuple[float,float]':
    '''
    A pair of conditionals (c(X)|b(X))[lb1, ub1] and (b(X)|a(X))[lb2, ub2]
    with a/1 probabilistic facts has a credal semantics only if 
    lb2 = 0 and ub2 <= 1,
    independently from the values of the bounds for the first conditional, or 
    lb2 >= 0 and ub2 = 1 
    and the first conditional follows the restrictions of Theorem 1.
    
    '''
    if (lb2 == 0 and ub2 <= 1) or (lb2 >=0 and ub2 == 1 and not (lb1 > 0 and ub1 < 1)):
        lp = 0
        up = 0
        
        for k in range(0, n_prob_facts):
            p = (prob**(k + 1)) * ((1 - prob)**(n_prob_facts - 1 - k))
            print(f"k = {k}, p = {p}, prob_facts = {prob}")
            lp += math.comb(n_prob_facts - 1, k) * p * delta(0,lb1,ub1,lb2,ub2,k,pedantic)
            up += math.comb(n_prob_facts - 1, k) * p * delta(1,lb1,ub1,lb2,ub2,k,pedantic)
    else:
        print("No credal semantics for this configuration.")
        sys.exit()

    return lp, up


if __name__ == "__main__":
    command_parser = argparse.ArgumentParser()
    command_parser.add_argument("lba", help="Lower bound", type=float)
    command_parser.add_argument("uba", help="Upper bound", type=float)
    command_parser.add_argument("lbb", help="Lower bound", type=float)
    command_parser.add_argument("ubb", help="Upper bound", type=float)
    command_parser.add_argument("n", help="Number of probabilistic facts", type=int)
    command_parser.add_argument("prob", help="Probaiblity", type=float)
    command_parser.add_argument("--pedantic", help="Pedantic mode", action="store_true")
    
    args = command_parser.parse_args()
    print(compute_probability(args.lba, args.uba, args.lbb, args.ubb, args.n, args.prob, args.pedantic))
