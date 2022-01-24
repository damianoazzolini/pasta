% There are 4 birds, of two types each:
% bird_a and bird_b (indexed both with 1 and 2).
% Birds of type a have an associated probability
% of 0.5 while birds of type a have an associated 
% probability of 0.4 
0.5::bird_a(1).
0.5::bird_a(2).
0.4::bird_b(1).
0.4::bird_b(2).

% A bird (of type a or b) can fly or not fly
fly_a(X);nofly_a(X):- bird_a(X).
fly_b(X);nofly_b(X):- bird_b(X).

% Constraint: at least 60% of the total birds fly
:-  #count{X:fly_a(X),bird_a(X)} = FBA,
    #count{X:fly_b(X),bird_b(X)} = FBB,
    FB = FBA + FBB,
    #count{X:bird_a(X)} = BA, 
    #count{X:bird_b(X)} = BB, 
    B = BA + BB,
    10*FB<6*B.

fly_1:- fly_a(1).
fly_1:- fly_b(1).

% Example query: fly_1
% What is the lower and the upper bound for the probability of
% the query fly_1?
% Expected: lower 0.21, upper: 0.52 