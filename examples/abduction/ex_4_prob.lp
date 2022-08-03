query:- b,c.
query:- d,e.
0.3::b.
abducible c.
0.6::d.
abducible e.

0.1:- c,e.

% Expected result
% Lower probability == upper probability for the query: 0.648
% Abductive explanations: 1
% {{c,e}}