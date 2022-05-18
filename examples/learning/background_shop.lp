#program('
bought(spaghetti) ; bought(steak) :- shops(john).
bought(spaghetti) ; bought(fish) :- shops(mary).
bought(spaghetti) ; bought(tuna) :- shops(carl).

t:- bought(spaghetti).
').

#target(t).

#learnable(shops(john)).
#learnable(shops(mary)).
#learnable(shops(carl)).

#train(1,2,3,4,5).
#test(3,4,5).

#pos_example(1,bought(fish)).
#pos_example(2,bought(spaghetti)).
#pos_example(3,bought(spaghetti)).
#pos_example(3,bought(fish)).
#neg_example(4,bought(tuna)).
#neg_example(5,bought(fish)).
#pos_example(6,bought(steak)).
#neg_example(7,bought(tuna)).






begin(model(3)).
bought(fish).
shops(mary).
end(model(3)).

begin(model(4)).
bought(fish).
shops(mary).
end(model(4)).

begin(model(5)).
bought(fish).
shops(mary).
end(model(5)).

begin(model(6)).
bought(fish).
shops(mary).
end(model(6)).

begin(model(7)).
bought(fish).
shops(mary).
end(model(7)).

begin(model(8)).
bought(fish).
shops(mary).
end(model(8)).

begin(model(9)).
bought(fish).
bought(steak).
shops(mary).
shops(john).
end(model(9)).

begin(model(10)).
bought(fish).
shops(mary).
end(model(10)).