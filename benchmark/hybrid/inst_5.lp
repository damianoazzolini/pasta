0.5::ad0.
0.5::ad2.

ac1:gaussian(0,1).
ac3:gaussian(0,1).

q0 ; q1 :- below(ac1, 0.5).
q0 ; q1 :- below(ac3, 0.5).

q0 :- below(ac1, 0.7), ad0.
q0 :- below(ac3, 0.7), ad2.

