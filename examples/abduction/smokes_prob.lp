abducible e(a,b).
abducible e(b,c).
abducible e(a,d).
abducible e(d,e).
abducible e(e,c).

0.5::fe(a,b).
0.5::fe(b,c).
0.5::fe(a,d).
0.5::fe(d,e).
0.5::fe(e,c).

friend(X,Y):- e(X,Y), fe(X,Y). 
friend(X,Y):- e(Y,X), fe(Y,X).

% :- friend(a,b).

smokes(b). 
% smokes(d).

smokes(X) ; no_smokes(X):- friend(X,Y), smokes(Y).

% query: smokes(c)
% Expected: 0 explanations