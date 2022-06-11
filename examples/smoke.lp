smokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

smokes(a).
smokes(c).
smokes(e).

friend(a,b).
friend(b,c).
friend(c,e).
friend(b,d).
friend(d,e).