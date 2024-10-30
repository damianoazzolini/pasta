win ; nwin :- red, green.
win :- blue, yellow.

map 0.4::red.
map 0.9::green.
map 0.1::blue.
map 0.6::yellow.

% $ pastasolver win_mpe.lp --map --query="win" --upper
% MPE: 0.1944
% MPE states: 1
% State 0: ['red', 'green', 'not blue', 'yellow']

% $ pastasolver win_mpe.lp --map --query="win"
% MPE: 0.032400000000000005
% MPE states: 1
% State 0: ['not red', 'green', 'blue', 'yellow']