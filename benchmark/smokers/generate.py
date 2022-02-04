#!/usr/bin/env python3

import networkx as nx

base = '''
smokes(X) :- smokesFact(X), smkAbd(X).
smokes(X) :- smokes(Y), influences(X,Y), infAbd(X,Y).

query(smokes(1)).
'''


def gen_smoker(n):
    for k in range(2, min(10, n)):
        fp_prob = open(f"smokers_{n}_{k}_prob.lp", 'w')
        fp_det = open(f"smokers_{n}_{k}_det.lp", 'w')
        for i in range(1, n + 1):
            fp_det.write(f"smokesFact({i}).\nabducible smkAbd({i}).\n")
            fp_prob.write(f"0.5::smokesFact({i}).\nabducible smkAbd({i}).\n")
        graph = nx.generators.random_graphs.barabasi_albert_graph(n, k)
        for i in range(1, n + 1):
            for j in range(1, n + 1):
                if graph.has_edge(i, j):
                    fp_det.write(f"influences({i},{j}).\nabducible infAbd({i},{j}).\n")
                    fp_prob.write(f"0.5::influences({i},{j}).\nabducible infAbd({i},{j}).\n")
        
        fp_prob.write(base)
        fp_det.write(base)
        
        fp_prob.close()
        fp_det.close()


if __name__ == "__main__":
    for i in range(3, 5):
        gen_smoker(i)
