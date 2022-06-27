map 0.5::bird(1).
map 0.5::bird(2).
map 0.5::bird(3).
map 0.5::bird(4).

fly(X);nofly(X):- bird(X).

:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.
