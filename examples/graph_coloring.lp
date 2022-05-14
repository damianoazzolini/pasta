color(X, red);color(X, blue);color(X, green) :- node(X).

node(0..3).

:- edge(X, Y ), col(C), color(X, C), color(Y, C).

col(red).
col(blue).
col(green).

color(0,blue).

0.5::edge(0, 1).
0.5::edge(1, 2).
0.5::edge(2, 0).
0.5::edge(1, 3).
0.5::edge(2, 3).