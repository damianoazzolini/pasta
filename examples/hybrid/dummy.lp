0.5::a.
0.2::f(a).
value:gaussian(0,1).
% 0.5::v0.
% 0.4207::v1.

qr:- a.
qr:- #constraint(value > 0.2).
% qr:- v1.
qr:- #constraint(value > 0), f(a), #constraint(value > 0.2).
% qr:- v0, f(a), v1.

% Exact value lp == up == 0.71035