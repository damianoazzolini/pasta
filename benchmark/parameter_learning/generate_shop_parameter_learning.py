import random
import clingo

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
cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.
:- cs(S), ce(C), 10* S < 4*C.
#show bought/1.
#show not_bought/1.
'''

lp4 = '''
#program('
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.
').

#learnable(shops(john)).
#learnable(shops(mary)).
#learnable(shops(carl)).
#learnable(shops(louis)).
'''

p8 = '''
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

bought(pizza,e) ; bought(nails,e) ; bought(onions,e) :- shops(e).
bought(socks,f) ; bought(beans,f) ; bought(nails,f)  :- shops(f).
bought(tomato,g) ; bought(tuna,g) ; bought(socks,g) :- shops(g).
bought(steak,h) ; bought(onions,h) ; bought(zucchini,f) :- shops(f).

bought(spaghetti):-  bought(spaghetti,_).
bought(steak):- bought(steak,_).
bought(tomato):- bought(tomato,_).
bought(tuna):- bought(tuna,_).
bought(onions):-  bought(onions,_).
bought(socks):-  bought(socks,_).
bought(pizza):-  bought(pizza,_).
bought(zucchini):-  bought(zucchini,_).

not_bought(spaghetti):- not bought(spaghetti,_).
not_bought(steak):- not bought(steak,_).
not_bought(tomato):- not bought(tomato,_).
not_bought(tuna):- not bought(tuna,_).
not_bought(onions):- not  bought(onions,_).
not_bought(pizza):- not  bought(pizza,_).
not_bought(socks):- not  bought(socks,_).
not_bought(zucchini):- not  bought(zucchini,_).

cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.


0{shops(john)}1.
0{shops(mary)}1.
0{shops(carl)}1.
0{shops(louis)}1.

0{shops(e)}1.
0{shops(f)}1.
0{shops(g)}1.
0{shops(h)}1.

#show bought/1.
#show not_bought/1.
'''

lp8 = '''
#program('
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

bought(pizza,e) ; bought(nails,e) ; bought(onions,e) :- shops(e).
bought(socks,f) ; bought(beans,f) ; bought(nails,f)  :- shops(f).
bought(tomato,g) ; bought(tuna,g) ; bought(socks,g) :- shops(g).
bought(steak,h) ; bought(onions,h) ; bought(zucchini,f) :- shops(f).

cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.
').

#learnable(shops(john)).
#learnable(shops(mary)).
#learnable(shops(carl)).
#learnable(shops(louis)).
#learnable(shops(e)).
#learnable(shops(f)).
#learnable(shops(g)).
#learnable(shops(h)).

'''

p10 = '''
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

bought(pizza,e) ; bought(nails,e) ; bought(onions,e) :- shops(e).
bought(socks,f) ; bought(beans,f) ; bought(nails,f)  :- shops(f).
bought(tomato,g) ; bought(tuna,g) ; bought(socks,g) :- shops(g).
bought(steak,h) ; bought(onions,h) ; bought(zucchini,h) :- shops(h).

bought(salami,i) ; bought(onions,i) ; bought(zucchini,i) ; bought(tape,i)  :- shops(i).
bought(nails,l) ; bought(tuna,l) ; bought(steak,l) ; bought(spaghetti,l)  :- shops(l).

bought(spaghetti):-  bought(spaghetti,_).
bought(steak):- bought(steak,_).
bought(tomato):- bought(tomato,_).
bought(tuna):- bought(tuna,_).
bought(onions):-  bought(onions,_).
bought(socks):-  bought(socks,_).
bought(pizza):-  bought(pizza,_).
bought(zucchini):-  bought(zucchini,_).
bought(salami):-  bought(salami,_).
bought(crayon):-  bought(crayon,_).
bought(tape):-  bought(tape,_).

not_bought(spaghetti):- not bought(spaghetti,_).
not_bought(steak):- not bought(steak,_).
not_bought(tomato):- not bought(tomato,_).
not_bought(tuna):- not bought(tuna,_).
not_bought(onions):- not  bought(onions,_).
not_bought(pizza):- not  bought(pizza,_).
not_bought(socks):- not  bought(socks,_).
not_bought(zucchini):- not  bought(zucchini,_).
not_bought(salami):- not  bought(salami,_).
not_bought(crayon):- not  bought(crayon,_).
not_bought(tape):- not  bought(tape,_).

cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.


0{shops(john)}1.
0{shops(mary)}1.
0{shops(carl)}1.
0{shops(louis)}1.

0{shops(e)}1.
0{shops(f)}1.
0{shops(g)}1.
0{shops(h)}1.

0{shops(i)}1.
0{shops(l)}1.


#show bought/1.
#show not_bought/1.
'''

lp10 = '''
#program('
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

bought(pizza,e) ; bought(nails,e) ; bought(onions,e) :- shops(e).
bought(socks,f) ; bought(beans,f) ; bought(nails,f)  :- shops(f).
bought(tomato,g) ; bought(tuna,g) ; bought(socks,g) :- shops(g).
bought(steak,h) ; bought(onions,h) ; bought(zucchini,f) :- shops(f).


bought(salami,i) ; bought(onions,i) ; bought(zucchini,i) ; bought(tape,i)  :- shops(i).
bought(nails,l) ; bought(tuna,l) ; bought(steak,l) ; bought(spaghetti,l)  :- shops(l).

cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.


').

#learnable(shops(john)).
#learnable(shops(mary)).
#learnable(shops(carl)).
#learnable(shops(louis)).
#learnable(shops(e)).
#learnable(shops(f)).
#learnable(shops(g)).
#learnable(shops(h)).
#learnable(shops(i)).
#learnable(shops(l)).

'''

p12 = '''
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

bought(pizza,e) ; bought(nails,e) ; bought(onions,e) :- shops(e).
bought(socks,f) ; bought(beans,f) ; bought(nails,f)  :- shops(f).
bought(tomato,g) ; bought(tuna,g) ; bought(socks,g) :- shops(g).
bought(steak,h) ; bought(onions,h) ; bought(zucchini,h) :- shops(h).

bought(salami,i) ; bought(onions,i) ; bought(zucchini,i) ; bought(tape,i)  :- shops(i).
bought(nails,l) ; bought(tuna,l) ; bought(steak,l) ; bought(spaghetti,l)  :- shops(l).

bought(beans,m) ; bought(onions,m) ; bought(steak,m) ; bought(spaghetti,m)  :- shops(m).
bought(nails,n) ; bought(tomato,n) ; bought(steak,n) ; bought(tuna,n)  :- shops(n).

bought(spaghetti):-  bought(spaghetti,_).
bought(steak):- bought(steak,_).
bought(tomato):- bought(tomato,_).
bought(tuna):- bought(tuna,_).
bought(onions):-  bought(onions,_).
bought(socks):-  bought(socks,_).
bought(pizza):-  bought(pizza,_).
bought(zucchini):-  bought(zucchini,_).
bought(salami):-  bought(salami,_).
bought(crayon):-  bought(crayon,_).
bought(tape):-  bought(tape,_).

not_bought(spaghetti):- not bought(spaghetti,_).
not_bought(steak):- not bought(steak,_).
not_bought(tomato):- not bought(tomato,_).
not_bought(tuna):- not bought(tuna,_).
not_bought(onions):- not  bought(onions,_).
not_bought(pizza):- not  bought(pizza,_).
not_bought(socks):- not  bought(socks,_).
not_bought(zucchini):- not  bought(zucchini,_).
not_bought(salami):- not  bought(salami,_).
not_bought(crayon):- not  bought(crayon,_).
not_bought(tape):- not  bought(tape,_).

cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.

0{shops(john)}1.
0{shops(mary)}1.
0{shops(carl)}1.
0{shops(louis)}1.

0{shops(e)}1.
0{shops(f)}1.
0{shops(g)}1.
0{shops(h)}1.

0{shops(i)}1.
0{shops(l)}1.

0{shops(m)}1.
0{shops(n)}1.

#show bought/1.
#show not_bought/1.

'''


lp12 = '''
#program('
bought(spaghetti,john) ; bought(steak,john) :- shops(john).
bought(spaghetti,mary) ; bought(beans,mary) :- shops(mary).
bought(tomato,carl) ; bought(tuna,carl) :- shops(carl).
bought(steak,louis) ; bought(onions,louis) :- shops(louis).

bought(pizza,e) ; bought(nails,e) ; bought(onions,e) :- shops(e).
bought(socks,f) ; bought(beans,f) ; bought(nails,f)  :- shops(f).
bought(tomato,g) ; bought(tuna,g) ; bought(socks,g) :- shops(g).
bought(steak,h) ; bought(onions,h) ; bought(zucchini,f) :- shops(f).


bought(salami,i) ; bought(onions,i) ; bought(zucchini,i) ; bought(tape,i)  :- shops(i).
bought(nails,l) ; bought(tuna,l) ; bought(steak,l) ; bought(spaghetti,l)  :- shops(l).

bought(beans,m) ; bought(onions,m) ; bought(steak,m) ; bought(spaghetti,m)  :- shops(m).
bought(nails,n) ; bought(tomato,n) ; bought(steak,n) ; bought(tuna,n)  :- shops(n).

cs(C):- #count{X : bought(spaghetti,X)} = C0, #count{X : bought(onions,X)} = C1, C = C0 + C1.
ce(C):- #count{X,Y : bought(Y,X)} = C.

:- cs(S), ce(C), 10* S < 4*C.

').

#learnable(shops(john)).
#learnable(shops(mary)).
#learnable(shops(carl)).
#learnable(shops(louis)).
#learnable(shops(e)).
#learnable(shops(f)).
#learnable(shops(g)).
#learnable(shops(h)).
#learnable(shops(i)).
#learnable(shops(l)).
#learnable(shops(m)).
#learnable(shops(n)).

'''

# # 20
# possible_objects = ["banana","pizza","cavoli","tomato","onion","piselli","crayon","salami","steak","spaghetti","garlic","rice","penne","bacon","tea","bag","socks","chair","peas","tape"]

# person_4 = ["a","b","c","d"]
# person_8 = ["a","b","c","d","e","f","g","h"]
# person_10 = ["a","b","c","d","e","f","g","h","i","l"]
# person_12 = ["a","b","c","d","e","f","g","h","i","l","m","n"]


def gen_shop_par_learning():

    size = 4
    program = p4
    lp = lp4

    ctl = clingo.Control(["0", "-Wnone", "--project"])
    ctl.add('base', [], program)
    ctl.ground([("base", [])])

    dataset = []

    with ctl.solve(yield_=True) as handle:
        for m in handle:
            dataset.append(str(m).split(' '))
            handle.get()

    # seleziono AS random
    init_examples = 100
    end_examples = 10000
    step_examples = 25
    for n_examples in range(init_examples, end_examples, step_examples):
        filename = f"bought_{size}_{n_examples}.lp"
        fp = open(f"{size}_increasing_int/" + filename, 'w')

        fp.write(lp)
        fp.write('\n')

        for i in range(0, n_examples):
            el = random.sample(dataset, 1)[0]
            # print(el)

            n_atoms = random.randrange(1, len(el) + 1)
            # print(n_atoms)
            atoms = random.sample(el, n_atoms)
            # print(atoms)
            for atom in atoms:
                if(atom.startswith('not_')):
                    print(
                        f"#negative({i},{atom.replace('not_','')}).", file=fp)
                else:
                    print(f"#positive({i},{atom}).", file=fp)

        s = "#train("
        n_test_set = int(n_examples/10)*4
        for i in range(0, n_examples - n_test_set):
            s = s + f'{i},'
        s = s[:-1] + ').\n'
        fp.write(s)

        s = "#test("
        for i in range(n_examples - n_test_set, n_examples):
            s = s + f'{i},'
        s = s[:-1] + ').\n'
        fp.write(s)

        fp.close()


if __name__ == "__main__":
    gen_shop_par_learning()
