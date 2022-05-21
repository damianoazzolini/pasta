#program('
pos:- circle(A), inside(B,A), pa.
pos:- circle(A), triangle(B), pb.
').

#learnable(pa).
#learnable(pb).

#train(1,2,3,4).

#positive(1,triangle(o5)).
#positive(1,circle(o3)).

#positive(2,triangle(o5)).
#positive(2,circle(o3)).

#negative(3,triangle(o5)).
#negative(3,circle(o3)).

#negative(4,triangle(o5)).
#negative(4,circle(o3)).
