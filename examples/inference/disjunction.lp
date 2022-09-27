% Since annotated disjunctive clauses are not yet supported, 
% this line
% 0.6::a;0.2::b.

% is transformed into
0.6::a1.
0.5::a2.
% where 0.5 = 0.2/(1 - 0.6)

a:- a1.
b:- not a1, a2.

f :- a.
{f} :- b.

query(f).

% Expected result
% Lower probability for the query: 0.6
% Upper probability for the query: 0.8