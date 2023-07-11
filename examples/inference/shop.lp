0.3::shops(a).
0.2::shops(b).
0.6::shops(c).

bought(beans,a) ; bought(spaghetti,a) :- shops(a).
bought(spaghetti,b) ; bought(steak,b) :- shops(b).
bought(tomato,c) ; bought(garlic,c) :- shops(c).
% bought(steak,d) ; bought(onions,d) :- shops(d).

cs(C):- #count{X:bought(spaghetti,X)} = C0, #count{X : bought(garlic,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 3*C.

qr:- bought(spaghetti,a).

% query: qr 
% Lower probability for the query: 0.0625
% Upper probability for the query: 0.5