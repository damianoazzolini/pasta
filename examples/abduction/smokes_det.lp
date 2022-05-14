abducible e(a,b).
abducible e(b,c).
abducible e(a,d).
abducible e(d,e).
abducible e(e,c).

friend(X,Y):- e(X,Y). 
friend(X,Y):- e(Y,X).

% :- friend(a,b).

smokes(b). 
smokes(d).

smokes(X) ; no_smokes(X):- friend(X,Y), smokes(Y).

:- #count{X:no_smokes(X)} = N, #count{X:smokes(X)} = S, 10*S < 8*(N+S).

% query: smokes(c)
% Expected: 
% Explanation 0
% ['e(b,c)']
% Explanation 1
% ['e(d,e)', 'e(e,c)']