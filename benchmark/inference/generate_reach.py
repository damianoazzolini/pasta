import sys
import math
import itertools
import argparse
import random

command_parser = argparse.ArgumentParser()
command_parser.add_argument("prob_facts_max", help="Number of probabilistic facts max",
                            type=int)
command_parser.add_argument("lb", help="Lower bound", type=float)
command_parser.add_argument("ub", help="Upper bound", type=float, default=1.0)
command_parser.add_argument("--asp", help="Probabilistic facts as\
    ASP choice rules", action="store_true", default=False)
command_parser.add_argument("--pasta", help="Print conditional", action="store_true",
                            default=False)

args = command_parser.parse_args()

n_prob_facts_max = args.prob_facts_max
lb = args.lb
ub = args.ub
asp_version = args.asp
pasta = args.pasta

if args.prob_facts_max < 5:
    n_prob_facts_max = 5

if pasta:
    asp_version = True

already_in = [[1, 2], [2, 3], [2, 4], [3, 5], [4, 5]]

random.seed(42)

for cf in range(5, n_prob_facts_max + 1):
    fp = open(f"reach_{cf}.lp", "w")
    for i in range(1, cf + 1):
        if asp_version and not pasta:
            print("{" + f"person({i})" + "}.", file=fp)
        else:
            print(f"0.4::person({i}).", file=fp)

    if asp_version:
        print("\nbuy(X):- reached(X), not not_buy(X). \nnot_buy(X):- reached(X), not buy(X).", file=fp)
    else:
        print("\nbuy(X):- reached(X), \+ not_buy(X). \nnot_buy(X):- reached(X), \+ buy(X).\
        \nquery(buy(1)).\n", file=fp)

    print("""
    advertise(1,2):- person(1), person(2).
    advertise(2,3):- person(2), person(3).
    advertise(2,4):- person(2), person(4).
    advertise(3,5):- person(3), person(5).
    advertise(4,5):- person(4), person(5).

    reach(A,B):- advertise(A,B).
    reach(A,B):- advertise(A,C), reach(C,B).

    reached(X):- person(X), reach(Y,X).
    reached(X):- person(X), advertise(X,Y).
    """, file=fp)

    for i in range(6, cf + 1):
        n0 = random.randint(1, cf)
        n1 = random.randint(1, cf)
        l = [n0, n1]
        l.sort()

        if l in already_in:
            n0 = n1

        while (n1 == n0):
            n1 = random.randint(1, cf)
            l = [n0, n1]
            l.sort()
            if l in already_in:
                n1 = n0

        a = l[0]
        b = l[1]
        print(f"advertise({a},{b}):- person({a}), person({b}).", file=fp)

    if pasta:
        print(":- #count{X:reached(X),buy(X)} = FB, #count{X:reached(X)} = B, 10*FB < "
              + f"{int(lb*10)}*B.", file=fp)
    else:
        # genero tutte le combinazioni di fatti prob
        already_gen: 'list[list[str]]' = []
        fbi_clauses_list: 'list[int]' = []

        for i in range(1, cf + 1):
            # case i small_chikens, only lb

            fb = math.ceil(i * float(lb))
            print(f"% {i} person: not fb < {fb} b", file=fp)
            # if fb not in already_gen:
            if fb > 0:
                # generate all the combinations
                comb = itertools.combinations(range(1, cf + 1), i)
                lfb: 'list[str]' = []
                lbrd: 'list[str]' = []
                for c in comb:
                    s_fb = f"fb{i}:- "
                    s_b = f"b{i}:- "
                    for el in c:
                        s_fb += f"buy({el}),"
                        s_b += f"reached({el}),"

                    s_fb = s_fb[:-1] + '.'
                    s_b = s_b[:-1] + '.'
                    lfb.append(s_fb)
                    lbrd.append(s_b)

                already_gen.append(lfb)
                if fb not in fbi_clauses_list:
                    fbi_clauses_list.append(fb)

                # if i == fb:
                # for s in lfb:
                #     print(s)
                # print('\n')

                for s in lbrd:
                    print(s, file=fp)
                print('\n', file=fp)

                if asp_version:
                    print(f":- b{i}, not fb{fb}.", file=fp)
                else:
                    print(f":- b{i}, \+ fb{fb}.", file=fp)

        for i in fbi_clauses_list:
            for cl in already_gen[i-1]:
                print(cl, file=fp)

    fp.close()
