import pasta
import math
import time


# FLY
prg_fly = '''
fly(X);nofly(X):- bird(X).
:- #count{X:fly(X),bird(X)} = FB, #count{X:bird(X)} = B, 10*FB<6*B.
'''
examples_fly = [['fly(1)','fly(2)','fly(3)','fly(4)'], ['fly(1)','nofly(2)','fly(3)'], ['fly(1)','fly(2)','fly(4)']]
pos_neg_fly = [0,1,1]
target_predicate_fly = "fly(1)"


# BONGARD
prg_bongard = '''
pos:- circle(A), inside(B,A), posa.
pos:- circle(A), triangle(B), posb.
% pos:- circle(A), posb.
'''

examples_bongard = [
    ['triangle(o5)', 'config(o5,up)', 'square(o4)', 'inside(o4,o5)', 'circle(o3)', 'triangle(o2)', 'config(o2,up)', 'inside(o2,o3)', 'triangle(o1)', 'config(o1,up)'],
    ['circle(o4)', 'circle(o3)', 'inside(o3,o4)', 'square(o2)', 'circle(o1)', 'inside(o1,o2)'],
    ['square(o3)', 'square(o2)', 'inside(o2,o3)', 'square(o1)'],
    ['triangle(o5)', 'config(o5,down)', 'triangle(o4)', 'config(o4,up)', 'inside(o4,o5)', 'circle(o3)', 'square(o2)', 'inside(o2,o3)', 'triangle(o1)', 'config(o1,up)'],
    ['circle(o4)', 'triangle(o3)', 'config(o3,up)', 'inside(o3,o4)', 'triangle(o2)', 'config(o2,down)', 'square(o1)', 'inside(o1,o2)']]
pos_neg_bongard = [1,0,0,1,1]


# SMOKES
examples_smokes = [["smokes(a)", "smokes(c)", "smokes(e)", "not_smokes(b)", "friend(a,b)", "friend(c,e)"], [
    "smokes(a)", "smokes(c)", "smokes(e)", "friend(c,e)", "friend(b,c)"], ["smokes(a)", "smokes(c)", "smokes(e)", "friend(b,d)"]]
pos_neg_smokes = [0,0,1]
prg_smokes = '''
smokes(Y) ; not_smokes(Y):- smokes(X), friend(X,Y).

:- #count{Y,X:smokes(X),friend(X,Y)} = F, #count{Y,X:smokes(X),friend(X,Y),smokes(Y)} = SF, 10*SF < 4*F.

0.7::smokes_w(d).

smokes(a).
smokes(c).
smokes(e).

smk:- smokes_w(d).
smk:- smokes(d).
'''
target_predicate_smokes = "smk"

# DUMMY
# examples_dummy = [["b","not a"],["b","not a"],["not b","not a"],["a"],["a","b"]]
examples_dummy = [["b"],["b"],["a"],["a","b"]]
# pos_neg_dummy = [0,1,0,1,1]
pos_neg_dummy = [0,1,1,1]
prg_dummy = '''
f :- a, c.
{f} :- not a, b, d.
'''
target_predicate_dummy = "f"

test_set_dummy = [["b"],["b"],["a"],["a","b"]]
pos_neg_test_set_dummy = [1,1,0,1]


# Cosa significano gli esempi negativi? Che la query è falsa in tutti
# i modelli in cui ci sono quegli esempi?

def generate_program_string(facts_prob : 'dict[str,float]', ex : 'list[str]', pos_neg : int, target_predicate : str, program : str) -> str:
    s = ""
    s = s + program
    to_assert = ""

    for e in ex:
        to_assert = to_assert + e + ".\n"

    for k in facts_prob:
        s = s + f"{facts_prob[k]}::{k}.\n"

    return s + to_assert + f"np:- not {target_predicate}.\n"


def compute_probability_example(facts_prob: 'dict[str,float]', example : 'list[str]', pos_neg : int, target_predicate : str, program : str, upper : bool) -> 'tuple[float,float]':
    s = generate_program_string(facts_prob, example, pos_neg, target_predicate, program)

    if pos_neg == 0:
        pasta_solver = pasta.Pasta("", "np", None)
    else:
        pasta_solver = pasta.Pasta("", target_predicate, None)
    
    lp, up, _ = pasta_solver.solve(from_string=s)
    
    return lp, up
    

# test


def test_results(test_set: 'list[list[str]]', pos_neg_test_set: 'list[int]', facts_prob: 'dict[str,float]', target_predicate: str, program: str) -> float:

    from sklearn.metrics import roc_auc_score
    import numpy as np
    from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
    import matplotlib.pyplot as plt
    # from sklearn.metrics import RocCurveDisplay

    probs_lp : list[float] = []
    probs_up : list[float] = []
    for i in range(0, len(test_set)):
        lp, up = compute_probability_example(facts_prob, test_set[i], pos_neg_test_set[i], target_predicate, program, upper)
        probs_lp.append(lp)
        probs_up.append(up)
    
    print(probs_lp)
    print(probs_up)
    
    pos_neg_test_set_np = np.asarray(pos_neg_test_set, dtype=np.int0)
    probs_lp_np = np.asarray(probs_lp, dtype=np.float32)
    probs_up_np = np.asarray(probs_up, dtype=np.float32)

    print(pos_neg_test_set_np)
    print(probs_up_np)

    auc_lp = roc_auc_score(pos_neg_test_set_np, probs_lp_np)
    auc_up = roc_auc_score(pos_neg_test_set_np, probs_up_np)

    print(auc_lp)
    print(auc_up)
    
    from sklearn import metrics
    fpr, tpr, thresholds = metrics.roc_curve(pos_neg_test_set, probs_lp_np)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    display.plot()
    plt.show()

    fpr, tpr, thresholds = metrics.roc_curve(pos_neg_test_set, probs_up_np)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    display.plot()
    plt.show()

    precision_lp, recall_lp, _ = precision_recall_curve(pos_neg_test_set, probs_lp_np)
    disp = PrecisionRecallDisplay(precision=precision_lp, recall=recall_lp)
    disp.plot()
    plt.show()

    precision_up, recall_up, _ = precision_recall_curve(pos_neg_test_set, probs_up_np)
    disp = PrecisionRecallDisplay(precision=precision_up, recall=recall_up)
    disp.plot()
    plt.show()


    return auc_lp, auc_up

def to_logprob(lp : float, up : float, upper : bool):    
    if upper:
        return math.log(float(up)) if float(up) != 0 else 0
    else:
        return math.log(float(lp)) if float(lp) != 0 else 0


def parse_input_learning(filename : str, from_string : str = ""):
    '''
    #example(pos,Id,'atom') where Id is the Id of the Answer set and atom is the correspondent atom
    #test(IdList)
    #train(IdList)
    #target('atom') where atom is the target predicate in the head of a rule
    #program('program') where program is a set of clauses
    #learnable(atom) where atom is a probabilistic fact with init probability 0.5
    '''
    lines : list[str] = []

    if filename == "":
        lines = from_string.split('\n')
    else:
        fp = open(filename,"r")
        lines = fp.readlines()
        fp.close()

    i = 0
    program = ""
    target = ""
    prob_facts_dict : dict[str,float] = dict()
    pos_examples_dict : dict[int,list[str]] = dict()
    neg_examples_dict : dict[int,list[str]] = dict()
    pos_neg_examples : list[int] = []
    
    training_set : list[list[str]] = []
    test_set : list[list[str]] = []
    pos_neg_training : list[int] = []
    pos_neg_test : list[int] = []

    train_ids : list[int] = []
    test_ids : list[int] = []

    while i < len(lines):
        if lines[i].startswith("#program('"):
            i = i + 1
            while(not (lines[i].startswith("')."))):
                program = program + lines[i]
                i = i + 1
        elif lines[i].startswith("#target("):
            ll = lines[i].split("#target(")
            target = ll[1].replace('\n','')[:-2]
            i = i + 1
        elif lines[i].startswith("#learnable("):
            ll = lines[i].split("#learnable(")
            name = ll[1].replace('\n','')[:-2]
            prob_facts_dict[name] = 0.5
            i = i + 1
        elif lines[i].startswith("#pos_example("):
            ll = lines[i].split("#pos_example(")
            number = ll[1].split(',')[0]
            atom = ll[1].replace('\n','')[len(number) + 1 : -2]
            if int(number) in pos_examples_dict.keys():
                pos_examples_dict[int(number)].append(atom)
            else:
                pos_examples_dict[int(number)] = [atom]
            pos_neg_examples.append(1)
            i = i + 1
        elif lines[i].startswith("#neg_example("):
            ll = lines[i].split("#neg_example(")
            number = ll[1].split(',')[0]
            atom = ll[1].replace('\n','')[len(number) + 1 : -2]
            if int(number) in neg_examples_dict.keys():
                neg_examples_dict[int(number)].append(atom)
            else:
                neg_examples_dict[int(number)] = [atom]
            pos_neg_examples.append(0)
            i = i + 1
        elif lines[i].startswith("#train("):
            ll = lines[i].split("#train(")
            train_ids = list(map(int, ll[1].replace('\n','')[:-2].split(',')))
            i = i + 1
        elif lines[i].startswith("#test("):
            ll = lines[i].split("#test(")
            test_ids = list(map(int, ll[1].replace('\n','')[:-2].split(',')))
            i = i + 1
        else:
            i = i + 1


    for el in pos_examples_dict:
        print(pos_examples_dict[el])

    for el in neg_examples_dict:
        print(neg_examples_dict[el])

    for id in train_ids:
        if id in pos_examples_dict:
            training_set.append(pos_examples_dict[int(id)])
            pos_neg_training.append(1)
        elif id in neg_examples_dict:    
            training_set.append(neg_examples_dict[int(id)])
            pos_neg_training.append(0)

    for id in test_ids:
        if id in pos_examples_dict:
            test_set.append(pos_examples_dict[int(id)])
            pos_neg_test.append(1)
        elif id in neg_examples_dict:
            test_set.append(neg_examples_dict[int(id)])
            pos_neg_test.append(0)

    print(training_set)
    print(test_set)
    print(target)
    print(test_ids)
    print(train_ids)
    print(program)

    return training_set, pos_neg_training, test_set, pos_neg_test, program, target, prob_facts_dict

def learn_parameters(training_set : 'list[list[str]]', pos_neg_training : 'list[int]', test_set : 'list[list[str]]', pos_neg_test : 'list[int]', program : str, target_predicate : str, prob_facts_dict : 'dict[str,float]', upper : bool = False, verbose : bool = False):
    # training_set = examples_dummy
    # pos_neg = pos_neg_dummy
    # program = prg_dummy
    # target_predicate = target_predicate_dummy
    # facts_prob['c'] = 0.1
    # facts_prob['d'] = 0.1

    # test_set = test_set_dummy
    # pos_neg_test_set = pos_neg_test_set_dummy

    start_time = time.time()

    ll0 = -10000000
    epsilon = 10e-5

    # compute negative LL
    p = 0
    for i in range(0, len(training_set)):
        lp, up = compute_probability_example(
            prob_facts_dict, training_set[i], pos_neg_training[i], target_predicate, program, upper)
        p = p + to_logprob(lp, up, upper)

    ll1 = -p

    # loop
    n_iterations = 0
    while abs(ll1 - ll0) > epsilon:
        n_iterations = n_iterations + 1
        print(f"ll0: {ll0} ll1: {ll1}")
        ll0 = ll1
        # fisso un fatto e calcolo la somma degli E per ogni esempio
        expected_dict = {}
        for key in prob_facts_dict:
            upper0 = 0
            upper1 = 0
            lower0 = 0
            lower1 = 0

            for i in range(0, len(training_set)):
                # sum_exected_val_true = 0
                # sum_exected_val_false = 0
                # compute expected val
                s = generate_program_string(
                    prob_facts_dict, training_set[i], pos_neg_training[i], target_predicate, program)

                # Expectation: compute E[C_i1 | e]
                e = target_predicate if pos_neg_training[i] > 0 else "np"

                pasta_solver = pasta.Pasta("", key, e)
                lp, up, _ = pasta_solver.solve(from_string=s)

                upper1 = upper1 + float(up)
                lower1 = lower1 + float(lp)

                # Expectation: compute E[C_i0 | e]
                pasta_solver = pasta.Pasta("", "nfp", e)
                s = s + f"nfp:- not {key}.\n"
                lp, up, _ = pasta_solver.solve(from_string=s)

                upper0 = upper0 + float(up)
                lower0 = lower0 + float(lp)

            expected_dict[key] = [lower0, lower1, upper0, upper1]

        # Maximization: update probabilities \sum E[C_i1|e] / \sum (E[C_i1|e] + E[C_i0|e])
        if verbose:
            print(expected_dict)
            print(prob_facts_dict)

        for k in prob_facts_dict:
            # per upper
            if upper:
                prob_facts_dict[k] = expected_dict[k][3] / \
                    (expected_dict[k][3] + expected_dict[k][2])
            # per lower
            else:
                prob_facts_dict[k] = expected_dict[k][1] / \
                    (expected_dict[k][0] + expected_dict[k][1])

        # Compute negative LL
        p = 0
        # for ex in examples:
        for i in range(0, len(training_set)):
            lp, up = compute_probability_example(
                prob_facts_dict, training_set[i], pos_neg_training[i], target_predicate, program, upper)
            p = p + to_logprob(lp, up, upper)

        ll1 = -p

    print(f"ll0: {ll0} ll1: {ll1} Iterations: {n_iterations}")
    print(prob_facts_dict)

    end_time = time.time() - start_time

    print(end_time)

    print(test_results(test_set, pos_neg_test,
                       prob_facts_dict, target_predicate, program))


if __name__ == "__main__":

    program = "background_bongard.lp"
    # program = "background_example.lp"

    tr, pn_tr, ts, pn_ts, program, target, prob_facts_dict = parse_input_learning(
        program)
    upper = True
    verbose = False
    start_time = time.time()
    learn_parameters(tr,pn_tr,ts,pn_ts,program,target,prob_facts_dict,upper,verbose)
    end_time = time.time() - start_time

    print(end_time)

    # import sys
    # sys.exit()

    # facts_prob : 'dict[str,float]' = {}



    # FLY
    # examples = examples_fly
    # pos_neg = pos_neg_fly
    # program = prg_fly

    # facts_prob['bird(1)'] = 0.5
    # facts_prob['bird(2)'] = 0.5
    # facts_prob['bird(3)'] = 0.5
    # facts_prob['bird(4)'] = 0.5
    # target_predicate = "fly(1)"

    # BONGARD
    # examples = examples_bongard
    # pos_neg = pos_neg_bongard
    # program = prg_bongard

    # facts_prob['posa'] = 0.1
    # facts_prob['posb'] = 0.1
    # target_predicate = "pos"

    # SMOKES
    # examples = examples_smokes
    # pos_neg = pos_neg_smokes
    # program = prg_smokes
    # target_predicate = target_predicate_smokes

    # facts_prob["friend(a,b)"] = 0.5
    # facts_prob["friend(b,c)"] = 0.5
    # facts_prob["friend(c,e)"] = 0.5
    # facts_prob["friend(b,d)"] = 0.5
    # facts_prob["friend(d,e)"] = 0.5

    # DUMMY

