0.3::f(a,b).
0.3::f(a,c).

friend(X,Y):- f(X,Y).
friend(X,Y):- f(Y,X).

smokes(b).

qry:- smokes(a).

0{smokes(X)}1 :- smokes(Y), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 2*F.

% obtained
% expected lp == up == 0.3