% from dtproblog

decision umbrella.
decision raincoat.

0.3::rain.
0.5::wind.

broken_umbrella :- rain,umbrella,wind.
dry :- rain, raincoat.
dry :- rain, umbrella, not broken_umbrella.
dry :- not rain.

% utility facts
utility(broken_umbrella,-40).
utility(raincoat,-20).
utility(umbrella,-2).
utility(dry,60).

% pasta temp.pl -dt
% Utility: 43.0
% Choice: ['umbrella', 'not raincoat']