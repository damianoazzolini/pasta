#program('
    smokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).

    :- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

    smokes(a).
    smokes(c).
    smokes(e).
').

#learnable(friend(a,b)).
#learnable(friend(b,c)).
#learnable(friend(c,e)).
#learnable(friend(b,d)).
#learnable(friend(d,e)).

#train(0,1,2,3,4,5,6,7,8,9).

#test(1,3,5).

#positive(0,friend(b,c)).
#positive(1,friend(c,e)).
#positive(2,smokes(e)).
#positive(3,smokes(c)).
#positive(3,friend(b,d)).
#positive(3,smokes(a)).
#positive(3,friend(d,e)).
#positive(4,smokes(e)).
#negative(4,smokes(d)).
#positive(4,smokes(a)).
#positive(4,smokes(b)).
#positive(4,friend(a,b)).
#positive(4,friend(b,d)).
#positive(4,smokes(c)).
#positive(5,smokes(e)).
#positive(5,friend(d,e)).
#positive(5,smokes(c)).
#positive(6,friend(b,c)).
#positive(6,friend(a,b)).
#positive(6,friend(c,e)).
#positive(6,smokes(e)).
#positive(6,smokes(a)).
#positive(6,smokes(c)).
#positive(6,smokes(b)).
#negative(6,smokes(d)).
#positive(6,friend(b,d)).
#positive(7,smokes(c)).
#positive(8,friend(a,b)).
#positive(8,smokes(c)).
#positive(8,smokes(e)).
#positive(8,friend(c,e)).
#positive(8,friend(b,d)).
#positive(8,smokes(d)).
#positive(8,smokes(b)).
#positive(9,friend(b,d)).
#positive(9,smokes(c)).
#negative(9,smokes(d)).
#positive(9,smokes(e)).
#positive(9,friend(a,b)).
#positive(9,smokes(a)).