0.4::chicken(1).
0.4::chicken(2).
0.4::chicken(3).
0.4::chicken(4).

peacock(1):- not chicken(1).
peacock(2):- not chicken(2).
peacock(3):- not chicken(3).
peacock(4):- not chicken(4).

(fly(X) | chicken(X))[0.4,1].
(fly(X) | peacock(X))[0.1,1].

% the previous two conditionals are equivalent to 
% :- #count{X:fly(X),chicken(X)} = FB, #count{X:chicken(X)} = B, 10*FB<4*B.
% :- #count{X:fly(X),peacock(X)} = FB, #count{X:peacock(X)} = B, 10*FB>6*B.
% 0{fly(X)}1:- chicken(X).
% 0{fly(X)}1:- peacock(X).


% Query: fly(1)
% Expected
% Lower probability for the query: 0.0864
% Upper probability for the query: 0.9616