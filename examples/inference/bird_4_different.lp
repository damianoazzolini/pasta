% There are 4 birds (indexed with 1,2,3, and 4), each one with a
% different probability
0.11::bird(1).
0.13::bird(2).
0.17::bird(3).
0.19::bird(4).

% A bird can fly or not fly
0{fly(X)}1 :- bird(X).

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.

% A query can also be specified into the program with query/1
query(fly(1)).

% Example query: fly(1)
% What is the lower and the upper bound of the probability for fly(1)?
% Expected: lower: 0.102, upper: 0.11