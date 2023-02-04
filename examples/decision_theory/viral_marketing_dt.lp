% There is uncertainty on the presence of some people
% identified with person(1), ..., person(5).
% For example, there is probability 0.1 that person(1)
% is present, 0.2 that person(2) is present, and so on.
0.1::person(1).
0.2::person(2).
0.3::person(3).
0.4::person(4).
0.5::person(5).

% If the message of a product reaches one person, 
% he/she will advertise it to every friend he/she knows.
advertise(1,2):- person(1), person(2), a(1,2).
advertise(2,3):- person(2), person(3), a(2,3).
advertise(2,4):- person(2), person(4), a(2,4).
advertise(3,5):- person(3), person(5), a(3,5).
advertise(4,5):- person(4), person(5), a(4,5).

decision a(1,2).
decision a(2,3).
decision a(2,4).
decision a(3,5).
decision a(4,5).

reach(A,B):- advertise(A,B).
reach(A,B):- advertise(A,C), reach(C,B).

reached(X):- person(X), reach(_,X).
reached(X):- person(X), advertise(X,_).

% Once reached, a person can or cannot buy the product.
0{buy(X)}1 :- reached(X).

% At least 70% of the people reached buy an item.
:- #count{X:reached(X),buy(X)} = RB, #count{X:reached(X)} = R, 10*RB < 7*R.

% utility facts
utility(a(1,2),-40).
utility(a(2,3),-20).
utility(a(2,4),-2).
utility(a(3,5),60).
utility(a(4,5),60).

utility(buy(1),5).
utility(buy(2),5).
utility(buy(3),5).
utility(buy(4),5).
utility(buy(5),5).

% python3 pasta_solver.py temp.pl -dt
% Utility: 123.2
% Choice: ['not a(1,2)', 'not a(2,3)', 'not a(2,4)', 'a(3,5)', 'a(4,5)']