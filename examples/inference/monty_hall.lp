% monty hall problem. From https://cplint.eu/example/inference/monty.swinb

% these probabilistic facts encode the probability that the prize is
% behind door 1, 2, or 3. They encode an annotated disjunctive clause.
0.333::prizef(1).
0.5::prizef(2).
prize(1):- prizef(1).
prize(2):- not prizef(1), prizef(2).
prize(3):- not prizef(1), not prizef(2).
% the prize is behind each door with probability 1/3

% the player selects door 1
select(1).

% Monty opens one door
open(1); open(2); open(3).

% The door that is opened does not have the prize behind
:- prize(X), open(X).

% The door that is opened was not selected by the player
:- select(X), open(X).

% Monty offers a choice to change door
choice(1); choice(2); choice(3).

% The chosen door has not been selected before
:- choice(X), select(X).

% The chosen door was not opened by Monty
:- choice(X), open(X).

% The player keeps his choice and wins if he has selected a door with the prize
win_keep:-
  select(X),
  prize(X).

% The player switches and wins if the prize is behind the door that he has
% not selected and that Monty did not open
win_switch:-
  choice(X),
  prize(X).

% query: win_switch
% Lower probability == upper probability for the query: 0.6666666666666667

% query: win_keep
% Lower probability == upper probability for the query: 0.3333333333333333