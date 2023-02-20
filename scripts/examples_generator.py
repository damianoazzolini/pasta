def gen_fly():
    import random

    n_examples = 100

    exs = []

    for i in range(1,n_examples+1):
        len = random.randint(1,4)
        ex = []
        for k in range(1,len+1):
            if random.random() > 0.5:
                ex.append(f'fly({k})')
            else:
                ex.append(f'nofly({k})')
        exs.append(ex)

    print(exs)

def gen_smokes_par_learning():
    import clingo
    program = '''
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

    ctl = clingo.Control(["0", "-Wnone", "--project"])
    ctl.add('base', [], program)
    ctl.ground([("base", [])])

    pos = []
    neg = []
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
    n_examples = 100
    import random

    for i in range(0,n_examples):
        el = random.sample(dataset,1)[0]
        # print(el)

        n_atoms = random.randrange(1,len(el) + 1)
        # print(n_atoms)
        atoms = random.sample(el,n_atoms)
        # print(atoms)
        for atom in atoms:
            if(atom.startswith('not_')):
                print(f"#negative({i},{atom.replace('not_','')}).")
            else:
                print(f"#positive({i},{atom}).")

    print("#train(")
    for i in range(0,n_examples):
        print(f'{i},', end='')
    print(')')

    print("#test(")
    for i in range(0,n_examples):
        print(f'{i},', end='')
    print(')')

def gen_test_bongard():
    filename = "temp_temp.tmp"
    f = open(filename,"r")
    lines = f.readlines()

    i = 0
    l_ids = []

    while i < len(lines):
        if lines[i].startswith('begin(model('):
            id = lines[i].split('begin(model(')[1][:-4]
            print(f"% {id}")
            l_ids.append(id)
            i = i + 1
            first = True
            while(not lines[i].startswith('end(model(')):
                if('pos' in lines[i]):
                    pass
                else:
                    if first:
                        l = lines[i][:-2]
                        print(f"#positive({id},{l}).")
                        first = False
                i = i + 1
        i = i+1
    
    print("#train(")
    for id in l_ids:
        print(f'{id},',end='')
    print(')')
    # for i in range(0,len(pos)):
    #     atoms = pos[i]
    #     for atom in atoms:
    #         if atom != "smk" and not atom.startswith("friend"):
    #             print(f"#pos_example({i},{atom}).")
    
    # i = 0
    # for i in range(0,len(neg)):
    #     atoms = neg[i]
    #     for atom in atoms:
    #         if atom != "smk" and not atom.startswith("friend"):
    #             print(f"#neg_example({i + len(pos)},{atom}).")

    # import random

    # n_ex = 60

    # ids = [random.randrange(0, len(pos) + len(neg)) for _ in range(n_ex)]

    # s = "#train("
    # for id in ids:
    #     s = s + str(id) + ","
    # s = s[:-1] + ').' 
    # print(s)

    # s = "#test("
    # for i in range(0,len(pos) + len(neg)):
    #     if not (i in ids):
    #         s = s + str(i) + ","
    # s = s[:-1] + ').' 
    # print(s)
    
if __name__ == "__main__":
    gen_smokes_par_learning()
    # gen_test_bongard()
