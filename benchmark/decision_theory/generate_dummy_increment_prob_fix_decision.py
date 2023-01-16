'''
Generates this type of programs with an increasing number
of decision atoms and a fixed number of probabilistic facts.

0.4::a.
0.4::b.
0.4::c.
0.4::d.

decision da.
decision db.

utility(qr,2).
utility(nqr,-12).

qr :- da, a.
qr ; nqr:- db, b.

qr :- da, c.
qr ; nqr:- db, b.

'''

# fix decision atoms to 2,5,10,15 and increase the nuber of 
# probabilistic facts
# fix the number of utility atoms to 2.

n_decision_atoms_max = [2,5,10,15]
n_prob_facts_max = 4

for n_decision_atoms in n_decision_atoms_max:
    for n_prob_facts in range(1, n_prob_facts_max):
        fp = open(f"dt_{n_prob_facts}_prob_facts_max_{n_decision_atoms}_decision_atoms.lp","w")
        
        # write the probabilistic facts
        for i in range(0, n_prob_facts):
            fp.write(f"0.3::a({i}).\n")
            
        fp.write('\n')
        
        # write the decision atoms
        for i in range(0, n_decision_atoms):
            fp.write(f"decision da({i}).\n")
            
        fp.write('\n')

        for ii in range(0, max(n_prob_facts,n_decision_atoms)):
            if ii % 2 == 0:
                fp.write(f"qr:- a({ii % n_prob_facts}), da({ii % n_decision_atoms}).\n")
            else:
                fp.write(f"qr ; nqr :- a({ii % n_prob_facts}), da({ii % n_decision_atoms}).\n")

        fp.write('\n')
        
        fp.write('utility(qr,2).\nutility(nqr,-12).\n')
        
        fp.close()