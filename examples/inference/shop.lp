0.5::shops(a).
0.5::shops(b).
0.5::shops(c).
0.5::shops(d).

bought(spaghetti,a) ; bought(steak,a) :- shops(a).
bought(spaghetti,b) ; bought(beans,b) :- shops(b).
bought(tomato,c) ; bought(tuna,c) :- shops(c).
bought(steak,d) ; bought(onions,d) :- shops(d).

cs(C):- #count{X:bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.

qr:- bought(spaghetti,a).