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

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X),fa(X)} = FB, #count{X:bird(X),fa(X)} = B, 10*FB<6*B.
% :- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.

% Query fly(1)
% Expected: lp == up == 0.5
% Abd = {fa(1)}

% if we add the constraint that at least three abducibles
% should be selected
% at least 3 abducibles
% :- C = #count{X : fa(X)}, C < 3.
% we get:
% Lower probability for the query: 0.375
% Upper probability for the query: 0.5
% Abductive explanations: 3
% Explanation 0
% ['fa(1)', 'fa(2)', 'fa(4)']
% Explanation 1
% ['fa(1)', 'fa(2)', 'fa(3)']
% Explanation 2
% ['fa(1)', 'fa(3)', 'fa(4)']