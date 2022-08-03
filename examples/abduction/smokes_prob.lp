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

smokes(b).
smokes(d).

smokes(X) ; no_smokes(X):- friend(X,Y), smokes(Y).

0.2:- #count{X:no_smokes(X)} = N, #count{X:smokes(X)} = S, 10*S<8*(N+S).

% query: smokes(c)
% Lower probability for the query smokes(c): 0.125
% Upper probability for the query smokes(c): 0.425
% Abductive explanations 1
% Explanation 0
% {{e(b,c), e(d,e), e(e,c)}}