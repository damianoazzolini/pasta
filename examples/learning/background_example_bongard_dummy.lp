#program('
pos:- circle(A), inside(B,A), pa.
pos:- circle(A), triangle(B), pb.
').

#target(pos).

#learnable(pa).
#learnable(pb).

#train(1,2,3,4).

#pos_example(1,triangle(o5)).
#pos_example(1,circle(o3)).

#pos_example(2,triangle(o5)).
#pos_example(2,circle(o3)).

#neg_example(3,triangle(o5)).
#neg_example(3,circle(o3)).

#neg_example(4,triangle(o5)).
#neg_example(4,circle(o3)).
