import random
import clingo


program_lp = '''
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
'''

program_5 = '''
smokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

smokes(a).
smokes(c).
smokes(e).

0{friend(a,b)}1.
0{friend(b,c)}1.
0{friend(c,e)}1.
0{friend(b,d)}1.
0{friend(d,e)}1.

#show smokes/1.
#show not_smokes/1.
#show friend/2.
'''


def gen_smokes_par_learning_increasing_rvs():

    facts = ["friend(a,d)", "friend(a,c)", "friend(b,e)", "friend(c,f)", "friend(f,e)", "friend(d,g)", "friend(g,e)",
             "friend(g,h)", "friend(h,e)", "friend(f,i)", "friend(i,e)", "friend(i,j)", "friend(j,h)", "friend(h,k)", "friend(j,k)"]
    learnable_facts = ["#learnable(friend(a,d))", "#learnable(friend(a,c))", "#learnable(friend(b,e))", "#learnable(friend(c,f))", "#learnable(friend(f,e))", "#learnable(friend(d,g))", "#learnable(friend(g,e))",
                       "#learnable(friend(g,h))", "#learnable(friend(h,e))", "#learnable(friend(f,i))", "#learnable(friend(i,e))", "#learnable(friend(i,j))", "#learnable(friend(j,h))", "#learnable(friend(h,k))", "#learnable(friend(j,k))"]

    for i in range(0, len(facts)):
        ctl = clingo.Control(["0", "-Wnone", "--project"])
        ctl.add('base', [], program_5)

        for j in range(0, i):
            ctl.add('base', [], facts[j] + ".")

        ctl.ground([("base", [])])

        dataset = []

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                dataset.append(str(m).split(' '))
                handle.get()

        # seleziono AS random
        n_examples = 100
        instance = 4
        for n_examples in range(100, 501, 100):
            filename = f"smoke_{i}_{n_examples}_i{instance}.lp"
            fp = open(f"smoke_learning/increasing_rvs/{filename}", "w")
            fp.write(program_lp)
            print(i)
            print(n_examples)
            for j in range(0, i):
                print(j)
                fp.write(learnable_facts[j] + ".\n")

            fp.write('\n')
            for jj in range(0, n_examples):
                el = random.sample(dataset, 1)[0]
                # print(el)

                n_atoms = random.randrange(1, len(el) + 1)
                # print(n_atoms)
                atoms = random.sample(el, n_atoms)
                # print(atoms)
                for atom in atoms:
                    if(atom.startswith('not_')):
                        print(
                            f"#negative({jj},{atom.replace('not_','')}).", file=fp)
                    else:
                        print(f"#positive({jj},{atom}).", file=fp)

            s = "#train("
            n_test_set = int(n_examples/10)*4
            for jj in range(0, n_examples - n_test_set):
                s = s + f'{jj},'
            s = s[:-1] + ').\n'
            fp.write(s)

            s = "#test("
            for jj in range(n_examples - n_test_set, n_examples):
                s = s + f'{jj},'
            s = s[:-1] + ').\n'
            fp.write(s)

            fp.close()


def gen_smokes_par_learning_increasing_interpretations():
    ctl = clingo.Control(["0", "-Wnone", "--project"])
    ctl.add('base', [], program_5)
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
    end_examples = 10000
    step_examples = 20
    for n_examples in range(init_examples, end_examples, step_examples):
        filename = f"smoke_5_{n_examples}.lp"
        fp = open(f"smoke_learning/5_rv_increasing_int/{filename}", "w")
        fp.write(program_lp)
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
    # gen_smokes_par_learning_increasing_rvs()
    gen_smokes_par_learning_increasing_interpretations()
