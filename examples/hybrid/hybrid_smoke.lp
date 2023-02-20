smokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, 
   #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 
   10*SF < 4*F.

smokes(a).
smokes(c).

price:gaussian(0,1).

0.1::friend(a,b).
0.1::friend(b,c).
0.1::friend(c,e).
0.1::friend(b,d).
0.1::friend(d,e).

% 0.69::price.
qr:- smokes(e), below(price, 0.5).
% qr:- smokes(e), below(price, 0.5).

% Query: qr
% Expected result: 
% Lower probability for the query: 0.06223162151466119
% Upper probability for the query: 0.06976856234254794