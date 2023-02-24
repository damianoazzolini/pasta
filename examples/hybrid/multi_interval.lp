0.4::b.
a:gaussian(0,1).

q0:- between(a,0.4,0.5).
q0:- below(a,0.7), b.

% query: q0
% Expected: lp == up == 0.3248