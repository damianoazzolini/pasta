0.3::a. 
0.4::b.

decision da. 
decision db.
utility(qr, 2). 
utility(nqr, -12).

:- db, a.

qr :- da , a.
qr ; nqr :- db , b.

% pasta dummy_c1.lp -dtn --normalize
% Warning: Naive decision theory solver, you should use -dt.
% Utility: [0.6000000000000001, 0.6000000000000001]
% Choice: ['da', 'not db']