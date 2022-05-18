0.1::burglary.
0.2::earthquake.
0.7::al(mary).
0.7::al(john).

person(mary).
person(john).
alarm:- burglary.
alarm:- earthquake.

calls(john) :- person(john), alarm, al(john).
calls(mary) :- person(mary), alarm, al(mary).

burglary.
alarm.
person(john).
person(mary).

query(calls(mary)).

% Expected: lp == up == 0.7