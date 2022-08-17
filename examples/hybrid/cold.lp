0.7::rain.
% < 7
% 0.8413::temperature.
0.9::weak(1).
0.9::weak(2).
0.9::weak(3).
0.9::weak(4).
temperature:gaussian(5,2).
person(1..4).
cold(P):- person(P), weak(P), rain, #constraint(temperature < 7).
% cold(P):- person(P), weak(P), #constraint(temperature < 1).

% :- #count{X:person(X)} = P, #count{X:cold(X), person(X)} = SP, 10*SP < 2*P.
at_least_one_cold:- #count{C : cold(C)} > 0.

% p3 pasta_solver.py ../../examples/hybrid/cold.lp --query="at_least_one_cold" --approximate --stop-if-inconsistent
% Lower probability == upper probability for the query: 0.599