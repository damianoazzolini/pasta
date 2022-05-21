#program('
bought0(spaghetti) ; bought(steak) :- shops(john).
bought1(spaghetti) ; bought(fish) :- shops(mary).
bought2(spaghetti) ; bought(tuna) :- shops(carl).

bought(spaghetti):- bought0(spaghetti).
bought(spaghetti):- bought1(spaghetti).
bought(spaghetti):- bought2(spaghetti).
').

#learnable(shops(john)).
#learnable(shops(mary)).
#learnable(shops(carl)).

#train(1,2,3,4,5,6,7,8).
% #train(3).
#test(3,4,5).

#positive(1,bought(fish)).
#negative(1,bought(tuna)).
#positive(2,bought(spaghetti)).
#positive(3,bought(spaghetti)).
#positive(3,bought(fish)).
#negative(4,bought(tuna)).
#negative(5,bought(fish)).
#positive(6,bought(steak)).
#positive(7,bought(tuna)).
#negative(8,bought(tuna)).