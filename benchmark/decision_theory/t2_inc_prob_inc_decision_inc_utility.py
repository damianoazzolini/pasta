'''
# Generates programs with an increasing number
# of decision atoms, probabilistic facts, and utility atoms.
# Experiment t3.

# With n = d = 4 and 4 utilities we get (the utilities
# of the decision atoms are random and so you may get
# different values):

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

utility(da(0),0).
utility(da(1),4).
utility(da(2),-10).
utility(da(3),-3).
'''

import random
import sys

if len(sys.argv) == 1:
    print("Specify the maximum number of atoms as argument")
    sys.exit()

atoms_sizes_max = int(sys.argv[1])

for atoms_size in range(2, atoms_sizes_max):
    for n_utilities in range(0, atoms_size + 1):
        fp = open(f"t2_{atoms_size}_pf_{atoms_size}_dt_{n_utilities}_u.lp","w")
        
        # write the probabilistic facts
        for i in range(0, atoms_size):
            fp.write(f"0.3::a({i}).\n")
            
        fp.write('\n')
        
        # write the decision atoms
        for i in range(0, atoms_size):
            fp.write(f"decision da({i}).\n")
            
        fp.write('\n')

        for ii in range(0, atoms_size):
            if ii % 2 == 0:
                fp.write(f"qr:- a({ii}), da({ii}).\n")
            else:
                fp.write(f"qr ; nqr :- a({ii}), da({ii}).\n")

        fp.write('\n')
        
        fp.write('utility(qr,2).\nutility(nqr,-12).\n')
        
        fp.write('\n')
        
        for u in range(n_utilities):
            fp.write(f"utility(da({u}),{random.randint(-10,10)}).\n")
        
        fp.close()