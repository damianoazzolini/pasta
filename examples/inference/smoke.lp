0{smokes(Y)}1 :- smokes(X), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

smokes(a).
smokes(c).
smokes(e).

0.3::friend(a,b).
0.3::friend(b,c).
0.3::friend(c,e).
0.3::friend(b,d).
0.3::friend(d,e).

qry:- smokes(d).

% query: qry
% Lower probability for the query: 0
% Upper probability for the query: 0.09