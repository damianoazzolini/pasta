maybe_smoke(X) ; not_smokes(X) :- friend(X,Y), smokes(Y).
smokes_case_3(X):- maybe_smoke(X), influenced(X).
smokes_case_1(X) :- smokes_p(X).

smokes(X):- smokes_case_3(X).
smokes(X):- smokes_case_1(X).

{maybe_cancer(X)}:- smokes(X).
cancer_case_3(X):- maybe_cancer(X), cancer(X).
cancer_case_1(X):- cancer_s(X).

cancer(X) :- cancer_case_3(X).
cancer(X) :- cancer_case_1(X).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

% smokes(a).
% smokes(b).
% smokes(c).

% qr:- not cancer(a).
qr:- not cancer(a).

0.5::smokes_p(a).
0.5::smokes_p(b).
0.5::smokes_p(c).

0.5::influenced(a).
0.5::influenced(b).
0.5::influenced(c).

0.5::cancer(b).
0.5::cancer(c).

0.5::cancer_s(a).
0.5::cancer_s(b).
0.5::cancer_s(c).

friend(a,b).
friend(b,a).
friend(b,c).
friend(c,b).
friend(a,c).
friend(c,a).

% query: qr
% This program is inconsistent.
% If you want to perform inference, use the flag --normalize
% Lower probability == upper probability for the query: 0.5