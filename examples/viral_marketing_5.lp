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
advertise(1,2):- person(1), person(2).
advertise(2,3):- person(2), person(3).
advertise(2,4):- person(2), person(4).
advertise(3,5):- person(3), person(5).
advertise(4,5):- person(4), person(5).

reach(A,B):- advertise(A,B).
reach(A,B):- advertise(A,C), reach(C,B).

reached(X):- person(X), reach(_,X).
reached(X):- person(X), advertise(X,_).

% Once reached, a person can or cannot buy the product.
buy(X);notbuy(X):- reached(X).

% At least 70% of the people reached buy an item.
:- #count{X:reached(X),buy(X)} = RB, #count{X:reached(X)} = R, 10*RB < 7*R.

% Example query: buy(5)
% What is the lower and the upper bound of the probability for fly(1)?
% Expected: lower: 0.273, upper: 0.29 (with precision 3)
% For this example, you can get more accurate results with precision 5