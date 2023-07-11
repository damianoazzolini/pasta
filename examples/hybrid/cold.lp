0.7::rain.
0.9::weak(1).
0.9::weak(2).
0.9::weak(3).
0.9::weak(4).
temperature:gaussian(5,2).

% 0.8413::temperature.
tb:- below(temperature, 7.0).

cold(1):- weak(1), rain, tb.
cold(2):- weak(2), rain, tb.
cold(3):- weak(3), rain, tb.
cold(4):- weak(4), rain, tb.

% :- #count{X:person(X)} = P, #count{X:cold(X), person(X)} = SP, 10*SP < 2*P.
at_least_one_cold:- #count{C : cold(C)} > 0.

% pasta cold.lp --query="at_least_one_cold"
% Lower probability == upper probability for the query: 0.58888