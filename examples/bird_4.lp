% There are 4 birds (indexed with 1,2,3, and 4), each one with 
% probability 0.5
0.5::bird(1..4).

% A bird can fly or not fly
fly(X);nofly(X):- bird(X).

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.