abducible a.
abducible b.
abducible c.
abducible d.

0.4::fa.
0.5::fb.
0.6::fc.
0.7::fd.

query:- a,fa.
query:- b,fb,c,fc.
query:- a,fa,d,fd.

% query(query).
% Expected result: {{a,b,c}}
% Prob = 0.58