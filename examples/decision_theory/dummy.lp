0.3::a. 
0.4::b.
decision da.
decision db.
utility(qr,2).
utility(nqr, -12).
qr :- da , a.
qr :- db , b, not nqr.
nqr :- db , b, not qr.

% pasta examples/decision_theory/dummy.lp -dtopt
% Lowest utility: -4.799999999999999
% Choice: ['not da', 'db']
% Highest utility: 1.16
% Choice: ['da', 'db']