0.8::shops(a).
0.5::shops(b).

decision target(a).
decision target(b).

buy(spaghetti,a) ; buy(steak,a) :- shops(a), target(a).
buy(spaghetti,b) ; buy(beans,b) :- shops(b), target(b).

utility(target(a),-2). utility(target(b),-2).
utility(buy(spaghetti,a),6). utility(buy(steak,a),-11).
utility(buy(spaghetti,b),-3). utility(buy(beans,b),7).

cs(C):- 
  #count{X : buy(spaghetti,X)} = C0, 
  #count{X : buy(beans,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : buy(Y,X)} = C.

:- cs(S), ce(C), 10*S < 4*C.

% pasta shop.pl -dtn
% Utility: [2.8, 2.8]
% Choice: ['target(a)', 'not target(b)']