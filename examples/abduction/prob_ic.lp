a:- b,c.
a:- d,e.
0.3::b.
abducible c.
0.6::d.
abducible e.

0.1:- c,e.

% query: a
% Expected
% Lower probability == upper probability for the query a: 0.648
% 1 abductive explanations 
% Explanation 0
% ['c', 'e']