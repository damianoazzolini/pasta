map 0.2::gold(1).
0.3::gold(2).
map 0.7::gold(3).

valuable(X) ; not_valuable(X):- gold(X).

:- #count{X:valuable(X), gold(X)} = FB, #count{X:gold(X)} = B, 10*FB < 6*B.

% pasta gold_map.lp --query="valuable(1)" --map
% MAP: 0.060000000000000005
% Map states: 1
% State 0: ['gold(1)', 'not gold(3)']

% pasta gold_map.lp --query="valuable(1)" --map --upper
% MAP: 0.13999999999999999
% Map states: 1
% State 0: ['gold(1)', 'gold(3)']