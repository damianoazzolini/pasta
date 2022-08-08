smokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

smokes(a).
smokes(c).

0.4::price_ok(b).
price_ok(e):gaussian(0,1).

0.1::friend(a,b).
0.1::friend(b,c).
0.1::friend(c,e).
0.1::friend(b,d).
0.1::friend(d,e).

qr:- smokes(e), price_ok(e).