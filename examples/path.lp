% Path in probabilistic graph
0.2::edge(1,2).
0.2::edge(1,3).
0.2::edge(1,4).
0.2::edge(2,3).
0.2::edge(2,4).
0.2::edge(3,4).

path(X,Y):- edge(X,Z), path(Z,Y).
path(X,Y):- edge(X,Y).

query(path(1,4)).

% Lower probability == upper probability for the query: 0.267