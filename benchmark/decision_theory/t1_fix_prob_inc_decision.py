'''
# Generates programs with an increasing number
# of decision atoms and a fixed number of probabilistic facts.
# Experiment t1.

# With n = 2 and d = 4 we get:

0.3::a(0).
0.3::a(1).

decision da(0).
decision da(1).
decision da(2).
decision da(3).

qr:- a(0), da(0).
qr ; nqr :- a(1), da(1).
qr:- a(0), da(2).
qr ; nqr :- a(1), da(3).

utility(qr,2).
utility(nqr,-12).

'''

# fix prob facts to 2,5,10,15 and increase the nuber of decision atoms
# fix the number of utility atoms to 2.

import sys

if len(sys.argv) == 1:
    print("Specify the maximum number of decision atoms as argument")
    sys.exit()

n_prob_facts = [2,5,10,15]
n_max_atoms = int(sys.argv[1])

for n in n_prob_facts:
    for n_decision_atoms in range(1, n_max_atoms):
        fp = open(f"t1_{n}_pf_{n_decision_atoms}_dt.lp","w")
        
        # write the probabilistic facts
        for i in range(0, n):
            fp.write(f"0.3::a({i}).\n")
            
        fp.write('\n')
        
        # write the decision atoms
        for i in range(0, n_decision_atoms):
            fp.write(f"decision da({i}).\n")
            
        fp.write('\n')

        for ii in range(0, max(n, n_decision_atoms)):
            if ii % 2 == 0:
                fp.write(f"qr:- a({ii % n}), da({ii % n_decision_atoms}).\n")
            else:
                fp.write(f"qr ; nqr :- a({ii % n}), da({ii % n_decision_atoms}).\n")

        fp.write('\n')
        
        fp.write('utility(qr,2).\nutility(nqr,-12).\n')
        
        fp.close()