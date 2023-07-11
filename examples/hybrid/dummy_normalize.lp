0.4::b.
a:gaussian(0,1).
q0 ; q1:- below(a,0.5).
q0:- below(a,0.7), b.

ne1 :- not e1.
e1:- b, below(a, 0.2).
:- b, below(a, 0.2).

% pasta test_hybrid.lp --query="q0" --normalize
% Lower probability for the query: 0.09307694498958728
% Upper probability for the query: 0.6330737873228345