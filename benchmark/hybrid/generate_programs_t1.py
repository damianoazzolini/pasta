import sys

'''
0.4::b.
a:gaussian(0,1).
q0 ; q1:- below(a,0.5).
q0:- below(a,0.7), b.

Every instance of size $n$ is characterized by $n/2$ discrete 
probabilistic facts $df_i$, $n/2$ continuos probabilistic 
facts $cf_i$ with gaussian distribution, $n/2$ rules \texttt
{q0 ; q1:- below(cfi, 0.5).}, one for each continuos fact 
$cf_i$, and $n/2$ rules \texttt{q0:- below(cfi, 0.7), dfi.}, 
one for each discrete fact $df_i$. 

Query: q0.
'''

if len(sys.argv) != 2:
    print("Usage: pytohn3 generate_programs_t1.py <max_size>")
    sys.exit()
    
prefix = "t1_inst"


for i in range(2, int(sys.argv[1]), 2):
    discrete_facts : 'list[str]' = []
    continuous_facts : 'list[str]' = []
    disj_clauses : 'list[str]' = []
    clauses : 'list[str]' = []
    
    for ii in range(0, i):
        if ii % 2 == 0:
            # add 0.5::ad{ii}.
            discrete_facts.append(f"0.5::ad{ii}.\n")
        else:
            # add ac{i}:gaussian(0,1).
            # add q0;q1 :- below(ac{ii}, 0.5).
            # add q0 :- below(ac{ii}, 0.7), ad{ii-1}.
            continuous_facts.append(f"ac{ii}:gaussian(0,1).\n")
            disj_clauses.append(f"q0 ; q1 :- below(ac{ii}, 0.5).\n")
            clauses.append(f"q0 :- below(ac{ii}, 0.7), ad{ii-1}.\n")
    
    fp = open(f"{prefix}_{i}.lp", "w")
    # if i % 2 != 0:
    #     discrete_facts = discrete_facts[:-1]
    
    for f in discrete_facts:
        fp.write(f)
    fp.write('\n')

    for f in continuous_facts:
        fp.write(f)
    fp.write('\n')

    for f in disj_clauses:
        fp.write(f)
    fp.write('\n')

    for f in clauses:
        fp.write(f)
    fp.write('\n')
    
    fp.close()
