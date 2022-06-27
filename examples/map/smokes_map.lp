map 0.5::e(a,b).
map 0.5::e(b,c).
map 0.5::e(a,d).
map 0.5::e(d,e).
map 0.5::e(e,c).

friend(X,Y):- e(X,Y). 
friend(X,Y):- e(Y,X).

% :- friend(a,b).

smokes(b). 
smokes(d).

smokes(X) ; no_smokes(X):- friend(X,Y), smokes(Y).

:- #count{X:no_smokes(X)} = N, #count{X:smokes(X)} = S, 10*S < 8*(N+S).