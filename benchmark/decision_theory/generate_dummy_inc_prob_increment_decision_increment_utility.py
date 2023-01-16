'''
Generates this type of programs with an increasing number
of decision atoms and a fixed number of probabilistic facts.

0.3::a. 0.4::b.

decision da.
decision db.

utility(qr,2).
utility(nqr,-12).

qr :- da, a.
qr ; nqr:- db, b.
qr :- dc, a.
qr ; nqr:- dd, b.

'''

# fix prob facts to 2,5,10,15 and increase the nuber of decision atoms
# and probabilistic facts
# fix the number of utility atoms to 2.

# n_max_atoms = 5
atoms_sizes_max = 4

for atoms_size in range(2, atoms_sizes_max):
    for n_utilities in range(0, atoms_size + 1):
        fp = open(f"dt_{atoms_size}_prob_facts_{atoms_size}_decision_atoms_{n_utilities}_utilities.lp","w")
        
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
            fp.write(f"utility(da({u}),1).\n")
        
        fp.close()