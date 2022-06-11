import random

p4 = '''
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

bought(spaghetti):-  bought(spaghetti,_).
bought(steak):- bought(steak,_).
bought(tomato):- bought(tomato,_).
bought(tuna):- bought(tuna,_).
bought(onions):-  bought(onions,_).

not_bought(spaghetti):- not bought(spaghetti,_).
not_bought(steak):- not bought(steak,_).
not_bought(tomato):- not bought(tomato,_).
not_bought(tuna):- not bought(tuna,_).
not_bought(onions):- not  bought(onions,_).

0{shops(john)}1.
0{shops(mary)}1.
0{shops(carl)}1.
0{shops(louis)}1.

#show bought/1.
#show not_bought/1.
'''

def gen_shop_par_learning():
    import clingo

    ctl = clingo.Control(["0", "-Wnone", "--project"])
    ctl.add('base', [], p4)
    ctl.ground([("base", [])])

    dataset = []

    with ctl.solve(yield_=True) as handle:
        for m in handle:
            dataset.append(str(m).split(' '))
            handle.get()

    # seleziono AS random
    init_examples = 100
    end_examples = 101
    step_examples = 25
    for n_examples in range(init_examples, end_examples, step_examples):
        filename = f"bought_4_{n_examples}.lp"
        fp = open(filename)

        for i in range(0, n_examples):
            el = random.sample(dataset, 1)[0]
            # print(el)

            n_atoms = random.randrange(1, len(el) + 1)
            # print(n_atoms)
            atoms = random.sample(el, n_atoms)
            # print(atoms)
            for atom in atoms:
                if(atom.startswith('not_')):
                    print(f"#negative({i},{atom.replace('not_','')}).")
                else:
                    print(f"#positive({i},{atom}).")

        print("#train(")
        n_test_set = 10
        for i in range(0, n_examples - n_test_set):
            print(f'{i},', end='')
        print(')')

        print("#test(")
        for i in range(n_examples - n_test_set, n_examples):
            print(f'{i},', end='')
        print(')')

if __name__ == "__main__":
    gen_shop_par_learning()
