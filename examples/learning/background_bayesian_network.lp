% From: https://dtai.cs.kuleuven.be/problog/tutorial/learning/01_bayes.html

#program('
0.2::earthquake.

alarm0 :- burglary, earthquake, p_alarm1.
alarm1 :- burglary, not earthquake, p_alarm2.
alarm2 :- not burglary, earthquake, p_alarm3.

alarm:- alarm0.
alarm:- alarm1.
alarm:- alarm2.
').

#learnable(burglary).
#learnable(p_alarm1).
#learnable(p_alarm2).
#learnable(p_alarm3).

#negative(1,burglary).
#negative(1,alarm).

#negative(2,earthquake).
#positive(2,alarm).
#positive(2,burglary).

#negative(3,burglary).

#train(1,2,3).
#test(1,2,3).

% Expected
% Iterations: 71
% {'burglary': 0.3333333333333333, 'p_alarm1': 0.5, 'p_alarm2': 0.9986666666666667, 'p_alarm3': 0.000}