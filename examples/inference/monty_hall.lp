% monty hall problem. From https://cplint.eu/example/inference/monty.swinb

% these prebabilistic facts encode the probability that the prize is
% behind door 1, 2, or 3. It encodes an annotated disjunctive clause.
0.333::prizef(1).
0.5::prizef(2).
prize(1):- prizef(1).
prize(2):- not prizef(1), prizef(2).
prize(2):- not prizef(1), not prizef(2).
% the prize is behind each door with probability 1/3

% Monty opens door 2 or door 3 if the prize is behind door 1.
1{open_door(2); open_door(3)}1:- prize(1).
% Monty opens door 2 if the prize is behind door 3.
open_door(2):- prize(3).
% Monty opens door 3 if the prize is behind door 2.
open_door(3):- prize(2).

% the player keeps his choice and wins if he has selected a door with the prize
win_keep:- prize(1).

% the player switches and wins if the prize is behind the door that he has
% not selected and that Monty did not open
win_switch:-
  prize(2),
  open_door(3).

win_switch:-
  prize(3),
  open_door(2).

% Note that if you change the probability distribution of Monty 
% selecting a door to open when the prize is behind the door 
% selected by the player, then the probability of winning by 
% switching remains the same

% query: win_switch"
% Lower probability == upper probability for the query: 0.6666666666666667

% query: win_keep"
% Lower probability == upper probability for the query: 0.3333333333333333