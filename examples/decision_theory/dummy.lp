0.3::a. 
0.4::b.

decision da. 
decision db.
utility(qr, 2). 
utility(nqr, -12).

qr :- da , a.
qr ; nqr :- db , b.

% python3 pasta_solver.py ../examples/decision_theory/dummy.lp -dtn
% Warning: Naive decision theory solver, you should use -dt.
% Utility: [0.6, 0.6]
% Choice: ['da', 'not db']