abducible a(1).
abducible b.
abducible c.
abducible d.

query:- a(1).
query:- b,c.
query:- a(1),d.

% query(query).
% Expected result: {{a},{b,c}}