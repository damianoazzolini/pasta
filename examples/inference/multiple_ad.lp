0.05 :: am1 ; 0.65 :: am2 ; 0.3 :: am3.

x1 :- am1.
x0 :- am2.
x0 ; x1 :- am3.

0.05 :: bm1 ; 0.65 :: bm2 ; 0.3 :: bm3.

y1 :- bm1.
y0 :- bm2.
y0 ; y1 :- bm3.

qr:- x1.
qr:- y1.

% pasta k.lp --query="qr"
% Lower probability for the query: 0.0975
% Upper probability for the query: 0.5774999999999999