% There are 4 birds (indexed with 1,2,3, and 4), each one with 
% probability 0.5
0.5::bird(1).
0.5::bird(2).
0.5::bird(3).
0.5::bird(4).

% A bird can fly or not fly
0{fly(X)}1 :- bird(X).

% Constraint: at least 60% of the birds fly
:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.

% A query can also be specified into the program with query/1
% query(fly(1)).

% Example query: fly(1)
% What is the lower and the upper bound of the probability for fly(1)?
% Expected: lower: 0.25, upper: 0.5

% Call: 
% pasta bird_4.lp --query="fly(1)"

% Probability of fly(1) given that fly(2) is true
% pasta bird_4.lp --query="fly(1)" --evidence="fly(2)"
% Lower probability for the query: 0.125
% Upper probability for the query: 0.5714285714285714

% Approximate inference
% pasta bird_4.lp --query="fly(1)" --approximate