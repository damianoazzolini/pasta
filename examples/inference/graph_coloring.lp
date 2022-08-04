red(X) ; green(X) ; blue(X) :- node(X).

e(X,Y) :- edge(X,Y).
e(Y,X) :- edge(Y,X).

node(1..6).

:- e(X,Y), red(X), red(Y).
:- e(X,Y), green(X), green(Y).
:- e(X,Y), blue(X), blue(Y).

red(1).
green(4).
green(6).

0.6::edge(1, 2).
0.1::edge(1, 3).
0.4::edge(2, 5).
0.3::edge(2, 6).
0.3::edge(3, 4).
0.8::edge(4, 5).
0.2::edge(5, 6).

qr:- blue(3).

% query: qr
% Expected result:
% Lower probability for the query: 0.03
% Upper probability for the query: 1.0