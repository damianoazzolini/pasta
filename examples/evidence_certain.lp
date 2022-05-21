a0 ; b :- v0.
a1 ; c :- v1.

0.5::v0.
0.5::v1.

a:- a0.
a:- a1.

ev:- a, c.

qr:- a.

% qr | e -> Expected: lp == up == 1
% Note that if the first two rules are
% a;b:- v0.
% a;c:- v1.
% The probability of the qr | e is 0