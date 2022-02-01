abducible a.
abducible b.
abducible c.
abducible d.

0.4::fa.
0.5::fb.
0.6::fc.
0.7::fd.

q:- a, fa.
q:- b,fb,c,fc.
q:- a,fa,d,fd.

% query(q).
% Expected result: {{a,b,c}}
% Prob = 0.58