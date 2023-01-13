'''
Generates this type of programs with an increasing number
of probabilistic facts and decision atoms.

0.3::a. 0.4::b.
decision da. decision db.
utility(qr,2). utility(nqr,-12).

qr :- da, a.
qr ; nqr:- db, b.
'''

# fix prob facts to 2,5,10,15 and increase the nuber of decision atoms
# and probabilistic facts
# fix the number of utility atoms to 2.

n_utilities = [2,5,10,15]
n_max_atoms = 4

for s in range(2, n_max_atoms):
    # for n_decision_atoms in range(1,30):
    fp = open(f"dt_{s}_prob_facts_{s}_decision_atoms.lp","w")
    
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