0.5::bird(1).
0.5::bird(2).
0.5::bird(3).
0.5::bird(4).

(fly(X) | bird(X))[0.6,1].

% Query: fly(1)
% Expected: lower 0.25, upper 0.5

% Call:
% pasta bird_4_cond_1.lp --query="fly(1)"