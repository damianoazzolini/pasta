0.3::a(0).
0.3::a(1).
0.3::a(2).
0.3::a(3).

decision da(0).
decision da(1).
decision da(2).
decision da(3).

qr:- a(0), da(0).
qr ; nqr :- a(1), da(1).
qr:- a(2), da(2).
qr ; nqr :- a(3), da(3).

utility(qr,2).
utility(nqr,-12).

utility(da(0),-3).
utility(da(1),2).
utility(da(2),8).

% pasta k.lp -dtopt
% Utility: [6.316, 11.313999999999998, 0]
% Choice: ['not da(0)', 'da(1)', 'da(2)', 'da(3)']
