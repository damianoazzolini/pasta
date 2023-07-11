0.3::a. 
0.4::b.

decision da. 
decision db.
utility(qr, 2). 
utility(nqr, -12).

qr :- da , a.
qr ; nqr :- db , b.

% pasta dummy.lp -dtn
% Warning: Naive decision theory solver, you should use -dt.
% Utility: [0.6, 0.6]
% Choice: ['da', 'not db']