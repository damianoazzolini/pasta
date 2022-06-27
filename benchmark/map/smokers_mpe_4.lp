
friend(X,Y):- e(X,Y).
friend(X,Y):- e(Y,X).

smokes(X) ; no_smokes(X):- 
  friend(X,Y), smokes(Y).

:- #count{X:no_smokes(X)} = N, 
   #count{X:smokes(X)} = S, 
   10*S < 8*(N+S).

smokes(0).
smokes(2).

map 0.5::e(0,1).
map 0.5::e(0,2).
map 0.5::e(0,3).
map 0.5::e(2,3).
