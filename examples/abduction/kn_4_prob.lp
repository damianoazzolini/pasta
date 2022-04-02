path(X,Y) :- e(X,Y).
path(X,Y) :- e(X,Z), path(Z,Y).

e(X,Y):- edge(X,Y), pf(X,Y).

0.5::pf(1,2).
0.5::pf(1,3).
0.5::pf(1,4).
0.5::pf(2,3).
0.5::pf(2,4).
0.5::pf(3,4).

abducible edge(1,2).
abducible edge(1,3).
abducible edge(2,3).
abducible edge(1,4).
abducible edge(2,4).
abducible edge(3,4).

query(path(1,4)).