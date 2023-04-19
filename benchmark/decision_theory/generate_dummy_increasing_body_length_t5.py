'''
% Example program of size 4.

decision da(0).
decision da(1).
decision da(2).
decision da(3).

0.3::a(0).
0.3::a(1).
0.3::a(2).
0.3::a(3).

qr:- a(0), da(0), a(2), da(2).
qr;nqr:- a(1), da(1), a(3), da(3).

utility(qr,2).
utility(nqr,-12).

% The following values are random, not always 2 as in this example
utility(da(0), 2).
utility(da(1), 2).
utility(da(2), 2).
utility(da(3), 2).

'''

import random

atoms_sizes_max = 5

for atoms_size in range(2, atoms_sizes_max):
    fp = open(f"dt_increasing_body_{atoms_size}.lp", "w")

    # write the probabilistic facts
    for i in range(0, atoms_size):
        fp.write(f"0.3::a({i}).\n")

    fp.write('\n')

    # write the decision atoms
    for i in range(0, atoms_size):
        fp.write(f"decision da({i}).\n")

    fp.write('\n')
    
    # write the utilities
    for i in range(0, atoms_size):
        fp.write(f"utility(da({i}),{random.randint(-10,10)}).\n")

    fp.write('\n')
    
    # body for qr
    pfe = ""
    # body for qr ; nqr
    pfo = ""

    for j in range(0, atoms_size, 2):
        pfe += f"a({j}), da({j}), "
    pfe = pfe[:-2]
    
    for j in range(1, atoms_size, 2):
        pfo += f"a({j}), da({j}), "
    pfo = pfo[:-2]
        
    fp.write(f"qr:- {pfe}.\n")
    fp.write(f"qr ; nqr :- {pfo}.\n")

    fp.write('\n')

    fp.write('utility(qr,2).\nutility(nqr,-12).\n')

    fp.write('\n')

    fp.close()
