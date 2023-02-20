diastolic_1_:gamma(70,1).
diastolic_2_:gamma(70,1).
diastolic_3_:gamma(70,1).
diastolic_4_:gamma(70,1).

systolic_1_:gamma(120,1).
systolic_2_:gamma(120,1).
systolic_3_:gamma(120,1).
systolic_4_:gamma(120,1).

problem_d(1):- outside(diastolic_1_, 60.0, 80.0).
problem_d(2):- outside(diastolic_2_, 60.0, 80.0).
problem_d(3):- outside(diastolic_3_, 60.0, 80.0).
problem_d(4):- outside(diastolic_4_, 60.0, 80.0).

problem_s(1):- outside(systolic_1_, 110.0, 130.0).
problem_s(2):- outside(systolic_2_, 110.0, 130.0).
problem_s(3):- outside(systolic_3_, 110.0, 130.0).
problem_s(4):- outside(systolic_4_, 110.0, 130.0).

problem(P):- problem_d(P).
problem(P):- problem_s(P).

stroke(P) ; not_stroke(P) :- problem(P).

:- #count{X:problem(X)} = P, #count{X:stroke(X), problem(X)} = SP, 10*SP < 4*P.

high_number_strokes:- CS = #count{X : stroke(X)}, CS > 1.

% ?- high_number_strokes.
% Lower probability for the query: 0.3246639370373975
% Upper probability for the query: 0.6994697831759257