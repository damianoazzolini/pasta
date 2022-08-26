% (f1(X)|h(X))[0.2,1].

0{f1(Y)}1:- h(Y).

:- #count{X:h(X)} = H,
   #count{X:f1(X),h(X)} = FH,
   10*FH < 2*H.

0{f2(X,Y)}1 :- h(Y), r(X,Y).

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

qr:- not f1(a), not f1(b), not f1(c), not f1(d), not f1(e), not f2(a,b), not f2(b,c), not f2(c,d), not f2(d,e).

qry:- not qr.

% qr:- f(0), f(0,1).

% Example query: qry.
% Expected: Lower probability == upper probability for the query qry: 0.605797
% Note that this program has some worlds without answer sets, so there
% is a probability loss.
% To normalize the probability, use the flags --no-minimal --normalize
% Lower probability == upper probability for the query: 0.7926756008980096