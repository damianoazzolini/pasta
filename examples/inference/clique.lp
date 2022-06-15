0.5::e(1, 2).
0.5::e(1, 4).
0.5::e(2, 1).
0.5::e(2, 2).
0.5::e(2, 4).
0.5::e(3, 3).

v(X) :- e(X,Y).
v(Y) :- e(X,Y).

2{in(X) : v(X)}.

:- in(X), in(Y), v(X), v(Y), X!=Y, not e(X,Y), not e(Y,X).

% Example query
% python3 pasta.py ../../examples/clique.lp --query="in(1)"
% Lower probability for the query in(1): 0.4375
% Upper probability for the query in(1): 0.875