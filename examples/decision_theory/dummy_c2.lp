0.3::a. 
0.4::b.

decision da. 
decision db.
utility(qr, 2). 
utility(nqr, -12).

:- a, b.

qr :- da , a.
qr ; nqr :- db , b.

% python3 pasta_solver.py ../examples/decision_theory/dummy_c2.lp -dtn --normalize
% Utility: [0.4090909090909091, 0.4090909090909091]
% Choice: ['da', 'not db']