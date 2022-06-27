#!/usr/bin/env python3

import networkx as nx  # type: ignore

base = '''
friend(X,Y):- e(X,Y).
friend(X,Y):- e(Y,X).

smokes(X) ; no_smokes(X):- 
  friend(X,Y), smokes(Y).

:- #count{X:no_smokes(X)} = N, 
   #count{X:smokes(X)} = S, 
   10*S < 8*(N+S).

'''

lower = 4
upper = 5
step = 1
attach_edges = 2

# MAP
for n_nodes in range(lower, upper, step):
    fp_map = open(f"smokers_map_{n_nodes}.lp", "w")
    fp_mpe = open(f"smokers_mpe_{n_nodes}.lp", "w")
    fp_mpe.write(base)
    fp_map.write(base)

    graph = nx.generators.random_graphs.barabasi_albert_graph(  # type: ignore
        n_nodes, attach_edges)
    for i in range(0, n_nodes):
        if i % 2 == 0:
            fp_map.write(f"smokes({i}).\n")
            fp_mpe.write(f"smokes({i}).\n")

    i = 0
    fp_mpe.write('\n')
    fp_map.write('\n')
    for a in graph.edges:  # type: ignore
        if i % 2 == 0:
            fp_map.write(f"0.5::e({str(a[0])},{str(a[1])}).\n")  # type: ignore
        else:
            fp_map.write(f"map 0.5::e({str(a[0])},{str(a[1])}).\n")  # type: ignore
        
        fp_mpe.write(f"map 0.5::e({str(a[0])},{str(a[1])}).\n")  # type: ignore

        i = i + 1

    fp_map.close()
    fp_mpe.close()
