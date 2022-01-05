% (f(X)|h(X))[0.2,1]

f1(Y) ; not_f1(Y):- h(Y).

:- #count{X:h(X)} = H,
   #count{X:f1(X),h(X)} = FH,
   10*FH < 2*H.

% (f(X, Y )|h(Y ),r(X, Y ))[0.9,1]

f2(X,Y) ; not_f2(X,Y) :- h(Y), r(X,Y).

:- #count{X:h(X),r(X,_)} = H,
   #count{Y,X:f2(X,Y),h(Y),r(X,Y)} = FH,
   10*FH < 9*H.

qry:- f1(A).
qry:- f2(A,B).

0.3::h(a).
0.3::h(b).
0.3::h(c).
0.3::h(d).
0.3::h(e).
0.3::r(a,b).
0.3::r(b,c).
0.3::r(c,d).
0.3::r(d,e).
0.3::r(e,f).