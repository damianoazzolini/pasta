% There are 10 birds (indexed with 1,2,...,10), each one with 
% probability 0.5
0.5::bird(1..10).

% A bird can fly or not fly
fly(X);nofly(X):- bird(X).

% Constraint: at least 80% of the birds fly
:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<8*B.

% Example query: fly(1)
% What is the lower and the upper bound of the probability for fly(1)?
% Expected: lower: 0.127, upper: 0.5

% Example query: nofly(1)
% What is the lower and the upper bound of the probability for nofly(1)?
% Expected: lower: 0, upper: 0.373