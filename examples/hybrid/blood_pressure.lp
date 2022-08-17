people(1..4).

diastolic(1..4):gaussian(70,10).
systolic(1..4):gaussian(120,10).

% 0.1587::p00.
% 0.1587::p01.
% 0.1587::p02.
% 0.1587::p03.

% 0.1587::p10.
% 0.1587::p11.
% 0.1587::p12.
% 0.1587::p13.

% 0.1587::p20.
% 0.1587::p21.
% 0.1587::p22.
% 0.1587::p23.

% 0.1587::p30.
% 0.1587::p31.
% 0.1587::p32.
% 0.1587::p33.

% this does not work since diastolic(I) is not ground
% problem_d(I):- people(I), #constraint(diastolic(I) < 60).
problem_d(1):- #constraint(diastolic(1) < 60).
problem_d(2):- #constraint(diastolic(2) < 60).
problem_d(3):- #constraint(diastolic(3) < 60).
problem_d(4):- #constraint(diastolic(4) < 60).
% problem_d(1):- p00.
% problem_d(2):- p01.
% problem_d(3):- p02.
% problem_d(4):- p03.

problem_d(1):- #constraint(diastolic(1) > 80).
problem_d(2):- #constraint(diastolic(2) > 80).
problem_d(3):- #constraint(diastolic(3) > 80).
problem_d(4):- #constraint(diastolic(4) > 80).
% problem_d(1):- p10.
% problem_d(2):- p11.
% problem_d(3):- p12.
% problem_d(4):- p13.

problem_s(1):- #constraint(systolic(1) < 110).
problem_s(2):- #constraint(systolic(2) < 110).
problem_s(3):- #constraint(systolic(3) < 110).
problem_s(4):- #constraint(systolic(4) < 110).
% problem_s(1):- p20.
% problem_s(2):- p21.
% problem_s(3):- p22.
% problem_s(4):- p23.

problem_s(1):- #constraint(systolic(1) > 130).
problem_s(2):- #constraint(systolic(2) > 130).
problem_s(3):- #constraint(systolic(3) > 130).
problem_s(4):- #constraint(systolic(4) > 130).
% problem_s(1):- p30.
% problem_s(2):- p31.
% problem_s(3):- p32.
% problem_s(4):- p33.

problem(P):- problem_d(P).
problem(P):- problem_s(P).

stroke(P) ; not_stroke(P) :- problem(P).

:- #count{X:problem(X)} = P, #count{X:stroke(X), problem(X)} = SP, 10*SP < 4*P.

high_number_strokes:- CS = #count{X : stroke(X)}, CS > 1.

% ?- high_number_strokes.
% Lower probability for the query: 0.3110605113964001
% Upper probability for the query: 0.6860577432422588