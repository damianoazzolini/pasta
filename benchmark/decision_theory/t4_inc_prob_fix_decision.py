'''
# Generates programs with an increasing number
# of probabilistic facts and a fixed number of decision atoms.
# Experiment t4.

# With n = 4 and d = 2 we get:

0.3::a(0).
0.3::a(1).
0.3::a(2).
0.3::a(3).

decision da(0).
decision da(1).

qr:- a(0), da(0).
qr ; nqr :- a(1), da(1).
qr:- a(2), da(0).
qr ; nqr :- a(3), da(1).

utility(qr,2).
utility(nqr,-12).

'''

# fix decision atoms to 2,5,10,15 and increase the nuber of
# probabilistic facts
# fix the number of utility atoms to 2.

import sys

if len(sys.argv) == 1:
    print("Specify the maximum number of probabilistic facts as argument")
    sys.exit()

n_decision_atoms_max = [2, 5, 10, 15]
n_prob_facts_max = int(sys.argv[1])

for n_decision_atoms in n_decision_atoms_max:
    for n_prob_facts in range(1, n_prob_facts_max):
        fp = open(f"t4_{n_decision_atoms}_d_{n_prob_facts}_pf.lp", "w")

        # write the probabilistic facts
        for i in range(0, n_prob_facts):
            fp.write(f"0.3::a({i}).\n")

        fp.write('\n')

        # write the decision atoms
        for i in range(0, n_decision_atoms):
            fp.write(f"decision da({i}).\n")

        fp.write('\n')

        for ii in range(0, max(n_prob_facts, n_decision_atoms)):
            if ii % 2 == 0:
                fp.write(
                    f"qr:- a({ii % n_prob_facts}), da({ii % n_decision_atoms}).\n")
            else:
                fp.write(
                    f"qr ; nqr :- a({ii % n_prob_facts}), da({ii % n_decision_atoms}).\n")

        fp.write('\n')

        fp.write('utility(qr,2).\nutility(nqr,-12).\n')

        fp.close()
