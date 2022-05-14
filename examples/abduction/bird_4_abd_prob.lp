% 4 probabilistic birds
0.5::bird(1).
0.5::bird(2).
0.5::bird(3).
0.5::bird(4).

% 4 abducibles
abducible fa(1).
abducible fa(2).
abducible fa(3).
abducible fa(4).

% A bird can fly or not fly
fly(X);nofly(X):- bird(X), fa(X).
% fly(X);nofly(X):- bird(X).

% at least 3 abducibles
:- C = #count{X : fa(X)}, C < 3.

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X),fa(X)} = FB, #count{X:bird(X),fa(X)} = B, 10*FB<6*B.
% :- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.

% Query fly(1)
% Expected: lp == up == 0.5
% Abd = {fa(1)}