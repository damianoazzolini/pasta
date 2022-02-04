0.5::bird(1).
0.5::bird(2).
0.5::bird(3).
% 0.5::bird(4).

abducible fa(1).
abducible fa(2).
abducible fa(3).

% A bird can fly or not fly
fly(X);nofly(X):- bird(X), fa(X).

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X),fa(X)} = FB, #count{X:bird(X),fa(X)} = B, 10*FB<6*B.

% Query fly(1)
% Expected: lp == up == 0.5
% Abd = {fa(1)}