people(1..4).

diastolic(1..4):gaussian(70,10).
systolic(1..4):gaussian(120,10).

problem_d(P):- #constraint(diastolic(P) < 60).
problem_d(P):- #constraint(diastolic(P) > 80).

problem_s(P):- #constraint(systolic(P) < 110).
problem_s(P):- #constraint(systolic(P) > 130).

problem(P):- problem_d(P).
problem(P):- problem_s(P).

stroke(P) ; not_stroke(P) :- problem(P).

:- #count{X:problem(X)} = P, 
   #count{X:stroke(X), problem(X)} = SP, 
   10*SP < 4*P.

high_number_strokes:- 
   CS = #count{X : stroke(X)}, CS > 1.

% ?- high_number_strokes.
% Lower probability for the query: 0.3650249557826206
% Upper probability for the query: 0.7365810555651996