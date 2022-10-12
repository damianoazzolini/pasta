map 0.5::e(a,b).
map 0.5::e(b,c).
map 0.5::e(a,d).
map 0.5::e(d,e).
map 0.5::e(e,c).

friend(X,Y):- e(X,Y). 
friend(X,Y):- e(Y,X).

% :- friend(a,b).

smokes(b). 
smokes(d).

smokes(X) ; no_smokes(X):- friend(X,Y), smokes(Y).

:- #count{X:no_smokes(X)} = N, #count{X:smokes(X)} = S, 10*S < 8*(N+S).

% > python3 pasta_solver.py smokes_map.pl --query="smokes(c)" --upper --map
% Warning: Brave (upper) MPE can be solved in a faster way using the --solver flag.
% MPE: 0.03125
% MPE states: 20
% State 0: ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'not e(e,c)']
% State 1: ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'not e(e,c)']
% State 2: ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'e(d,e)', 'not e(e,c)']
% State 3: ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'not e(e,c)']
% State 4: ['e(a,b)', 'e(b,c)', 'e(a,d)', 'e(d,e)', 'not e(e,c)']
% State 5: ['e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'not e(e,c)']
% State 6: ['e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'not e(e,c)']
% State 7: ['e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'not e(e,c)']
% State 8: ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'e(d,e)', 'e(e,c)']
% State 9: ['not e(a,b)', 'not e(b,c)', 'e(a,d)', 'e(d,e)', 'e(e,c)']
% State 10: ['not e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'e(e,c)']
% State 11: ['not e(a,b)', 'not e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)']
% State 12: ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)']
% State 13: ['not e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'e(e,c)']
% State 14: ['e(a,b)', 'not e(b,c)', 'e(a,d)', 'e(d,e)', 'e(e,c)']
% State 15: ['e(a,b)', 'not e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)']
% State 16: ['e(a,b)', 'e(b,c)', 'e(a,d)', 'e(d,e)', 'e(e,c)']
% State 17: ['e(a,b)', 'e(b,c)', 'not e(a,d)', 'e(d,e)', 'e(e,c)']
% State 18: ['e(a,b)', 'e(b,c)', 'e(a,d)', 'not e(d,e)', 'e(e,c)']
% State 19: ['e(a,b)', 'e(b,c)', 'not e(a,d)', 'not e(d,e)', 'e(e,c)']