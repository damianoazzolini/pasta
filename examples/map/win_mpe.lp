win :- red, green.
win :- blue, yellow.

map 0.4::red.
map 0.9::green.
map 0.5::blue.
map 0.6::yellow.

% query = win
% expected: 
% MPE: 0.162
% ['not red', 'green', 'blue', 'yellow']