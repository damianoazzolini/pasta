0.3::f(a,b).
0.3::f(a,c).

friend(X,Y):- f(X,Y).
friend(X,Y):- f(Y,X).

smokes(b).

qry:- smokes(a).

smokes(X) ; not_smokes(X):- smokes(Y), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 2*F.

% obtained
% Lower probability for the query: 0.3
% Upper probability for the query: 0.51
% expected lp == up == 0.3