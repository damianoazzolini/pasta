from fileinput import filename
import random


def gen_smokes_par_learning():
    import clingo
    program = '''
    smokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).

    :- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

    smokes(a).
    smokes(c).
    smokes(e).

    0{friend(a,b);not_friend(a,b)}1.
    0{friend(b,c);not_friend(b,c)}1.
    0{friend(c,e);not_friend(c,e)}1.
    0{friend(b,d);not_friend(b,d)}1.
    0{friend(d,e);not_friend(d,e)}1.

    #show smokes/1.
    #show not_smokes/1.
    #show friend/2.
    '''

    ctl = clingo.Control(["0", "-Wnone", "--project"])
    ctl.add('base', [], program)
    ctl.ground([("base", [])])


    dataset = []

    with ctl.solve(yield_=True) as handle:
        for m in handle:
            dataset.append(str(m).split(' '))
            # if "smk" in str(m):
            #     pos.append(str(m).split(' '))
            #     # print('pos')
            # else:
            #     neg.append(str(m).split(' '))
            #     # print('neg')
            # # print(m)
            handle.get()

    # seleziono AS random
    init_examples = 100
    end_examples = 101
    step_examples = 25
    for n_examples in range(init_examples, end_examples, step_examples):
        # filename = f"smoke_5_{n_examples}.lp"
        # fp = open()

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


# def gen_test_bongard():
#     filename = "temp_temp.tmp"
#     f = open(filename, "r")
#     lines = f.readlines()

#     i = 0
#     l_ids = []

#     while i < len(lines):
#         if lines[i].startswith('begin(model('):
#             id = lines[i].split('begin(model(')[1][:-4]
#             print(f"% {id}")
#             l_ids.append(id)
#             i = i + 1
#             first = True
#             while(not lines[i].startswith('end(model(')):
#                 if('pos' in lines[i]):
#                     pass
#                 else:
#                     if first:
#                         l = lines[i][:-2]
#                         print(f"#positive({id},{l}).")
#                         first = False
#                 i = i + 1
#         i = i+1

#     print("#train(")
#     for id in l_ids:
#         print(f'{id},', end='')
#     print(')')


if __name__ == "__main__":
    gen_smokes_par_learning()
