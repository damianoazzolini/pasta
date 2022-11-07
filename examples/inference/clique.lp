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

% Example query: in(1)
% Expected:
% Error: found 4 worlds without answer sets: [0, 1, 4, 5]
% 000000{ }
% 000001{ e(3,3) }
% 000100{ e(2,2) }
% 000101{ e(2,2) e(3,3) }
% This program has some worlds without answer sets, so it is
% inconsistent.
% If you want to fix this, use the flags --normalize
% Lower probability for the query: 0.4666666666666667
% Upper probability for the query: 0.9333333333333333