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

n_prob_facts = [2,5,10,15]
n_max_atoms = 5

for n in n_prob_facts:
    for n_decision_atoms in range(1, n_max_atoms):
        fp = open(f"dt_{n}_prob_facts_max_{n_decision_atoms}_decision_atoms.lp","w")
        
        # write the probabilistic facts
        for i in range(0, n):
            fp.write(f"0.3::a({i}).\n")
            
        fp.write('\n')
        
        # write the decision atoms
        for i in range(0, n_decision_atoms):
            fp.write(f"decision da({i}).\n")
            
        fp.write('\n')

        for ii in range(0, n_decision_atoms):
            if ii % 2 == 0:
                fp.write(f"qr:- a({ii % n}), da({ii}).\n")
            else:
                fp.write(f"qr ; nqr :- a({ii % n}), da({ii}).\n")

        fp.write('\n')
        
        fp.write('utility(qr,2).\nutility(nqr,-12).\n')
        
        fp.close()