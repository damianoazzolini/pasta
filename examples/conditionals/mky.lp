% (f1(X)|h(X))[0.2,1].

f1(Y) ; not_f1(Y):- h(Y).

:- #count{X:h(X)} = H,
   #count{X:f1(X),h(X)} = FH,
   10*FH < 2*H.

f2(X,Y) ; not_f2(X,Y) :- h(Y), r(X,Y).

:- #count{X:h(X),r(X,_)} = H,
   #count{Y,X:f2(X,Y),h(Y),r(X,Y)} = FH,
   10*FH < 1*H.

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

% qry:- f1(a).
% qry:- f1(b).
% qry:- f1(c).
% qry:- f1(d).
% qry:- f1(e).
% qry:- f2(a,b).
% qry:- f2(b,c).
% qry:- f2(c,d).
% qry:- f2(d,e).

qr:- not f1(a), not f1(b), not f1(c), not f1(d), not f1(e), not f2(a,b), not f2(b,c), not f2(c,d), not f2(d,e).

qry:- not qr.

% qr:- f(0), f(0,1).

% Example query: qry.
% python3 pasta.py mky.lp --query="qry"
% Expected: Lower probability == upper probability for the query qry: 0.605797