map 0.5::bird(1).
map 0.5::bird(2).
map 0.5::bird(3).
map 0.5::bird(4).

fly(X);nofly(X):- bird(X).

:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.

% python3 pasta_solver.py ../examples/map/bird_4_map.lp  --map --query="fly(1)"
% MPE: 0.0625
% MPE states: 4
% State 0: ['bird(1)', 'bird(2)', 'not bird(3)', 'not bird(4)']
% State 1: ['bird(1)', 'not bird(2)', 'bird(3)', 'not bird(4)']
% State 2: ['bird(1)', 'not bird(2)', 'not bird(3)', 'bird(4)']
% State 3: ['bird(1)', 'not bird(2)', 'not bird(3)', 'not bird(4)']