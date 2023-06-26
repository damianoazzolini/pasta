'''
# Generates programs with an increasing number
# of probabilistic facts.

# With n = 4 we get:

0.3::a(0).
0.3::a(1).
0.3::a(2).
0.3::a(3).

qr:- a(0).
qr ; nqr :- a(1).
qr:- a(2).
qr ; nqr :- a(3).

The second version is
0.3::a(0). 0.3::a(1).
0.3::a(2). 0.3::a(3).
qr:- a(0), a(2).
qr :- a(1), a(3), not nqr.
nqr :- a(1), a(3), not qr.
'''

# fix prob facts to 2,5,10,15 and increase the nuber of decision atoms
# fix the number of utility atoms to 2.

import argparse

command_parser = argparse.ArgumentParser()
command_parser.add_argument("min_n", help="Min number of probabilistic facts", type=int)
command_parser.add_argument("max_n", help="Max number of probabilistic facts", type=int)
command_parser.add_argument("--sm", help="SMProbLog version", action="store_true")
command_parser.add_argument("--v2", help="Second version", action="store_true")

args = command_parser.parse_args()

for n in range(args.min_n, args.max_n + 1):
    fp = open(f"t1_{n}_pf.lp", "w")

    # write the probabilistic facts
    for i in range(0, n):
        fp.write(f"0.3::a({i}).\n")

    fp.write('\n')
    
    if args.v2:
        sqr = "qr:- "
        snqr = ""
        for ii in range(0, n):
            if ii % 2 == 0:
                sqr += f"a({ii % n}), "
            else:
                snqr += f"a({ii % n}), "
        fp.write(f"{sqr[:-2]}.\n")
        if args.sm:
            fp.write(f"qr:- {snqr[:-2]}, \+ nqr.\n")
            fp.write(f"nqr:- {snqr[:-2]}, \+ qr.\n")
        else:
            fp.write(f"qr ; nqr:- {snqr[:-2]}.\n") 
    else:
        for ii in range(0, n):
            if ii % 2 == 0:
                fp.write(f"qr:- a({ii % n}).\n")
            else:
                if args.sm:
                    fp.write(f"qr :- a({ii % n}), \+ nqr.\nnqr :- a({ii % n}), \+ qr.\n")
                else:
                    fp.write(f"qr ; nqr :- a({ii % n}).\n")

        fp.write('\n')

    if args.sm:
        fp.write("\nquery(qr).\n")
    fp.close()
