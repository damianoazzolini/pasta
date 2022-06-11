#program('
    maybe_smoke(X) ; not_smokes(X) :- friend(X,Y), smokes(Y).
    smokes_case_3(X):- maybe_smoke(X), influenced(X).
    smokes_case_1(X) :- smokes_p(X).

    smokes(X):- smokes_case_3(X).
    smokes(X):- smokes_case_1(X).

    maybe_cancer(X) ; not_cancer(X):- smokes(X).
    cancer_case_3(X):- maybe_cancer(X), cancer(X).
    cancer_case_1(X):- cancer_s(X).

    cancer(X) :- cancer_case_3(X).
    cancer(X) :- cancer_case_1(X).

    :- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

    friend(a,b).
    friend(b,a).
    friend(b,c).
    friend(c,b).
    friend(a,c).
    friend(c,a).
').

#learnable(smokes_p(a)).
#learnable(smokes_p(b)).
#learnable(smokes_p(c)).

#learnable(influenced(a)).
#learnable(influenced(b)).
#learnable(influenced(c)).

#learnable(cancer(a)).
#learnable(cancer(b)).
#learnable(cancer(c)).

#learnable(cancer_s(a)).
#learnable(cancer_s(b)).
#learnable(cancer_s(c)).

#train(0,1,2,3,4,5,6,7,8,9).

#positive(0,cancer_s(a)).
#positive(0,cancer(b)).
#positive(1,cancer(a)).
#positive(1,cancer(b)).
#positive(1,smokes_p(c)).
#positive(2,smokes_p(a)).
#positive(2,influenced(a)).
#positive(2,smokes_p(c)).
#positive(2,cancer(c)).
#positive(3,cancer(b)).
#positive(3,influenced(a)).
#positive(3,cancer(a)).
#positive(4,smokes_p(c)).
#positive(5,smokes_p(a)).
#positive(5,influenced(a)).
#positive(6,cancer(b)).
#positive(7,cancer(a)).
#positive(7,influenced(a)).
#positive(7,influenced(b)).
#positive(7,cancer_s(a)).
#positive(7,cancer(c)).
#positive(8,cancer(b)).
#positive(8,cancer(a)).
#positive(8,smokes_p(b)).
#positive(9,smokes_p(a)).