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
% Lower probability for the query in(1): 0.4375
% Upper probability for the query in(1): 0.875
% This program has some worlds without answer sets, so there
% can be a probability loss.
% If you want to fix this, use the flags --no-minimal --normalize
% Lower probability for the query: 0.4666666666666667
% Upper probability for the query: 0.9333333333333333