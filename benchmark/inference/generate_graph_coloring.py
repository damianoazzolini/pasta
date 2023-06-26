'''
# Generates programs with an increasing number of probabilistic facts
# in a graph. Graph coloring task.
'''

import argparse
import random

command_parser = argparse.ArgumentParser()
# command_parser.add_argument("min_n", help="Min number of nodes", type=int)
command_parser.add_argument("max_n", help="Max number of nodes", type=int)

args = command_parser.parse_args()

random.seed(1)

for n in range(6, args.max_n + 1):
    fp_pasta = open(f"pasta_gc_{n}_pf.lp", "w")
    fp_sm = open(f"sm_gc_{n}_pf.lp", "w")

    preamble = """
red(X) :- node(X), not green(X), not blue(X).
green(X) :- node(X), not red(X), not blue(X).
blue(X) :- node(X), not red(X), not green(X).

e(X,Y) :- edge(X,Y).
e(Y,X) :- edge(Y,X).

:- e(X,Y), red(X), red(Y).
:- e(X,Y), green(X), green(Y).
:- e(X,Y), blue(X), blue(Y).

0.6::edge(1, 2).
0.1::edge(1, 3).
0.4::edge(2, 5).
0.3::edge(2, 6).
0.3::edge(3, 4).
0.8::edge(4, 5).
0.2::edge(5, 6).

qr:- blue(3).

red(1).
green(4).
green(6).


"""

    preamble_sm = preamble.replace('not', '\+')

    fp_pasta.write(preamble)
    fp_sm.write(preamble_sm)
    fp_sm.write("\nquery(qr).\n")

    already_in = [[1, 2], [1, 3], [2, 5], [2, 6], [3, 4], [4, 5], [5, 6]]

    # write the probabilistic facts
    for i in range(0, n - 6 + 1):
        n0 = random.randint(1, n)
        n1 = random.randint(1, n)
        l = [n0, n1]
        l.sort()

        if l in already_in:
            n0 = n1

        while (n1 == n0):
            n1 = random.randint(1, n)
            l = [n0, n1]
            l.sort()
            if l in already_in:
                n1 = n0

        already_in.append(l)

        fp_pasta.write(f"0.3::edge({l[0]},{l[1]}).\n")
        fp_sm.write(f"0.3::edge({l[0]},{l[1]}).\n")

    flat_list = [item for sublist in already_in for item in sublist]
    for j in range(1, max(flat_list) + 1):
        fp_pasta.write(f"node({j}).\n")
        fp_sm.write(f"node({j}).\n")

    fp_pasta.write('\n')
    fp_sm.write('\n')

    fp_pasta.close()
    fp_sm.close()
