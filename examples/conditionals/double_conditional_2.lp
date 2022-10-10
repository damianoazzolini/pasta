0.4::chicken(1).
0.4::chicken(2).
0.4::chicken(3).
0.4::chicken(4).

0.6::peacock(1).
0.6::peacock(2).
0.6::peacock(3).
0.6::peacock(4).

(fly(X) | chicken(X))[0.4,1].
(fly(X) | peacock(X))[0,0.6].

% the previous two conditionals are equivalent to 
% :- #count{X:fly(X),chicken(X)} = FB, #count{X:chicken(X)} = B, 10*FB<4*B.
% :- #count{X:fly(X),peacock(X)} = FB, #count{X:peacock(X)} = B, 10*FB>6*B.
% 0{fly(X)}1:- chicken(X).
% 0{fly(X)}1:- peacock(X).


% Query: fly(1)
% Expected
% Lower probability for the query: 0.11736576000000004
% Upper probability for the query: 0.6220671999999997