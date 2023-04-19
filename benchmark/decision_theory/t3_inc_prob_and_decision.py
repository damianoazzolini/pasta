'''
# Generates programs with an increasing number
# of decision atoms and probabilistic facts.
# Experiment t3.

# With n = d = 4 we get:

0.3::a(0).
0.3::a(1).
0.3::a(2).
0.3::a(3).

decision da(0).
decision da(1).
decision da(2).
decision da(3).

qr:- a(0), da(0).
qr ; nqr :- a(1), da(1).
qr:- a(2), da(2).
qr ; nqr :- a(3), da(3).

utility(qr,2).
utility(nqr,-12).
'''

import sys

if len(sys.argv) == 1:
    print("Specify the maximum number of decision atoms as argument")
    sys.exit()

n_max_atoms = int(sys.argv[1])

for s in range(2, n_max_atoms):
    # for n_decision_atoms in range(1,30):
    fp = open(f"t3_{s}_pf_{s}_dt.lp","w")
    
    # write the probabilistic facts
    for i in range(0, s):
        fp.write(f"0.3::a({i}).\n")
    
    fp.write('\n')
    
    for i in range(0, s):
        fp.write(f"decision da({i}).\n")
    
    fp.write('\n')


    # write the decision atoms
    for ii in range(0, s):
        if ii % 2 == 0:
            fp.write(f"qr:- a({ii}), da({ii}).\n")
        else:
            fp.write(f"qr ; nqr :- a({ii}), da({ii}).\n")
    
    fp.write('\n')
    
    fp.write('utility(qr,2).\nutility(nqr,-12).\n')
        
    fp.close()