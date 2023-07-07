% 0.9::edge(a,b).
% 0.8::edge(b,c).
% 0.8::edge(b,d).
% 0.3::edge(c,e).
% 0.8::edge(d,e).

0.6::edge(a,b).
0.2::edge(b,c).
0.95::edge(d,e).

0.9::edge(c,f).
0.9::edge(a,d).
0.9::edge(e,f).

node(a).
node(b).
node(c).
node(d).
node(e).
node(f).

path(X,X):- node(X).
path(X,Y):- path(X,Z), edge(Z,Y).

{transmit(A,B)}:- path(A,B), node(A), node(B).

:- #count{A,B:transmit(A,B),path(A,B)} = RB, #count{A,B:path(A,B)} = R, 100*RB < 94*R.

qr:- transmit(a,f).

% query: transmit(a,e)
% Lower probability for the query: 0.15839999999999999
% Upper probability for the query: 0.6537600000000001