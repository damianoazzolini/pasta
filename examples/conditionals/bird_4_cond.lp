% there are 4 birds
bird(1).
bird(2).
bird(3).
bird(4).

% the bird indexed by 1 flies with probability 0.7
0.7::fly_w(1).

% The following two lines are equivalent to the conditional
% (fly(X) | bird(X))[0.6]

% A bird can fly or not fly
fly(X) ; not_fly(X):- bird(X).

% At least 60% of the birds flies
:- #count{X:fly(X),bird(X)} = FB, 
   #count{X:bird(X)} = B, 
   10*FB < 6*B.

fly:- fly(1).
fly:- fly_w(1).

% Query: fly(1)
% Expected: lower 0.7, upper 1.0
% That is, there are 2 worlds, 1 where fly(1) (the probabilistic
% fact) is true, the other where it is false
% The world with fly(1) contributes to both lower and upper 
% probability with 0.7
% For the other world, we need to consider the models: it has
% 5 and the query is true in 4 of them, so we have a contribution
% of 0.3 only to the upper probability.
% Overall: upper 0.7 + 0.3 = 1, lower = 0.7 
