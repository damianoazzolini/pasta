#!/usr/bin/env python3

import networkx as nx  # type: ignore

base = '''
smokes(X) ; no_smokes(X) :- smokesFact(X).
smokes(X) ; no_smokes(X) :- smokes(Y), influences(X,Y).

query(smokes(1)).

'''

if __name__ == "__main__":

    deterministic = True
    if deterministic:
        lower = 10
        upper = 1000
        step = 10
    else:
        lower = 4
        upper = 30
        step = 1

    attach_edges = 2
    for n_nodes in range(lower, upper, step):
        if deterministic is False:
            fp = open("prob/smokers_prob_" + str(n_nodes) + ".lp", 'w')
        else:
            fp = open("det/smokers_det_" + str(n_nodes) + ".lp", 'w')
        fp.write(base)

        graph = nx.generators.random_graphs.barabasi_albert_graph(  # type: ignore
            n_nodes, attach_edges)
        for i in range(1, n_nodes + 1):
            if i % 2 == 0:
                if deterministic is False:
                    fp.write(f"0.5::smokesFact({i}).\n")
                else:
                    fp.write(f"abducible smokesFact({i}).\n")

            else:
                fp.write(f"smokesFact({i}).\n")

        i = 0
        for a in graph.edges:  # type: ignore
            # fp_det.write(f"influences(" + str(a[0]) + "," + str(a[1]) + ").\nabducible infAbd("+ str(a[0]) + "," + str(a[1]) + ").\n")
            if i % 2 == 0:
                fp.write(f"abducible influences({str(a[0])},{str(a[1])}).\n")  # type: ignore
            else:
                fp.write(f"influences({str(a[0])},{str(a[1])}).\n")  # type: ignore

            i = i + 1

        fp.close()
