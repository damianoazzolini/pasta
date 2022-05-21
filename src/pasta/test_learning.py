from bleach import clean
import pasta
import math
import time

import sys

import re


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

interpretation_string = "interpretation"


# Cosa significano gli esempi negativi? Che la query è falsa in tutti
# i modelli in cui ci sono quegli esempi?

def generate_program_string(
    facts_prob : 'dict[str,float]', 
    atoms : 'list[str]', 
    program : str) -> str:
    
    s = ""
    s = s + program
    to_assert = f"{interpretation_string}:- "

    for e in atoms:
        to_assert = to_assert + e + ", "
    to_assert = to_assert[:-2] + ".\n"

    for k in facts_prob:
        s = s + f"{facts_prob[k]}::{k}.\n"

    return s + to_assert +"\n"


def get_tuple_regex(facts_prob : 'dict[str,float]'):
    l = []
    for el in facts_prob.keys():
        e = el.replace('(','').replace(')','')
        l.append(e)
        l.append("not_" + e)
    return tuple(l)


# FIXME: i need to perform these super complicated string manipulations
# since the worlds ids are strings with the name of the facts, rather
# than 01 strings
def compute_probability_interpretation(
        facts_prob : 'dict[str,float]',
        example : 'list[str]',
        program : str,
        index : int,
        interpretations_to_worlds: 'dict[int,list[tuple[str,int,int]]]') -> 'tuple[float,float]':

    if index not in interpretations_to_worlds:
        s = generate_program_string(facts_prob, example, program)

        # print("--- program")
        # print(example)
        # print(s)

        pasta_solver = pasta.Pasta("", interpretation_string, None)
        up : float = 0
        lp : float = 0
        lp, up = pasta_solver.inference(from_string=s)

        # print(pasta_solver.interface.model_handler.worlds_dict)
        # print(get_tuple_regex(facts_prob))
        # print('---')
        for w in pasta_solver.interface.model_handler.worlds_dict:
            el = pasta_solver.interface.model_handler.worlds_dict[w]
            # print(el)
            if index not in interpretations_to_worlds: 
                interpretations_to_worlds[index] = [[
                    w, 
                    1 if el.model_not_query_count == 0 and el.model_query_count > 0 else 0,
                    1 if el.model_query_count > 0 else 0
                ]]
            else:
                interpretations_to_worlds[index].append([
                    w, 
                    1 if el.model_not_query_count == 0 and el.model_query_count > 0 else 0,
                    1 if el.model_query_count > 0 else 0
                    ])
        # print(facts_prob)

        # print('---- DICT')
        # for el in interpretations_to_worlds:
        #     print(interpretations_to_worlds[el])


        # import sys
        # sys.exit()
    else:
        # l'interpretazione è nel dizionario, calcolo la probabilità
        # partendo da la stringa
        # print('found')
        lp = 0
        up = 0
        worlds_list = interpretations_to_worlds[index]
        for world in worlds_list:
            id = world[0]
            lpw = world[1]
            upw = world[2]
            delimiters = get_tuple_regex(facts_prob)
            regexPattern = '|'.join(map(re.escape, delimiters))
            id_w_list = re.findall(regexPattern, id)

            lp_contribution = 1
            up_contribution = 1
            for f_w in id_w_list:
                # identifico il fatto
                prob = -1
                for fact in facts_prob:
                    clean_fact = fact.replace('(','').replace(')','')
                    if clean_fact == f_w:
                        prob = facts_prob[fact]
                        break
                    elif "not_" + clean_fact == f_w:
                        prob = 1 - facts_prob[fact]
                        break
                # calcolo il contributo
                lp_contribution = lp_contribution * prob
                up_contribution = up_contribution * prob

            lp = lp + lp_contribution * lpw
            up = up + up_contribution * upw

    print(lp)
    print(up)
    return lp, up
    

# test


def test_results(test_set: 'list[list[str]]', pos_neg_test_set: 'list[int]', facts_prob: 'dict[str,float]', target_predicate: str, program: str) -> float:

    from sklearn.metrics import roc_auc_score
    import numpy as np
    # from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import RocCurveDisplay

    probs_lp : list[float] = []
    probs_up : list[float] = []
    ll = 0
    for i in range(0, len(test_set)):
        lp, up = compute_probability_interpretation(facts_prob, test_set[i], pos_neg_test_set[i], target_predicate, program, upper)
        probs_lp.append(lp)
        probs_up.append(up)
        ll = ll + to_logprob(lp, up, upper)
    
    print(f"Negative LL: {ll}")
    print(f"Probs lp: {probs_lp}")
    print(f"Probs up: {probs_up}")
    
    pos_neg_test_set_np = np.asarray(pos_neg_test_set, dtype=np.int0)
    probs_lp_np = np.asarray(probs_lp, dtype=np.float32)
    probs_up_np = np.asarray(probs_up, dtype=np.float32)

    # print(pos_neg_test_set_np)
    # print(probs_up_np)

    auc_lp = roc_auc_score(pos_neg_test_set_np, probs_lp_np)
    auc_up = roc_auc_score(pos_neg_test_set_np, probs_up_np)

    print(f"auc_lp: {auc_lp}")
    print(f"auc up: {auc_up}")
    
    # from sklearn import metrics
    # fpr, tpr, thresholds = metrics.roc_curve(pos_neg_test_set, probs_lp_np)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    # display.plot()
    # plt.show()

    # fpr, tpr, thresholds = metrics.roc_curve(pos_neg_test_set, probs_up_np)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    # display.plot()
    # plt.show()

    # precision_lp, recall_lp, _ = precision_recall_curve(pos_neg_test_set, probs_lp_np)
    # disp = PrecisionRecallDisplay(precision=precision_lp, recall=recall_lp)
    # disp.plot()
    # plt.show()

    # precision_up, recall_up, _ = precision_recall_curve(pos_neg_test_set, probs_up_np)
    # disp = PrecisionRecallDisplay(precision=precision_up, recall=recall_up)
    # disp.plot()
    # plt.show()


    return auc_lp, auc_up


def to_logprob(lp : float, up : float, upper : bool) -> float:    
    if upper:
        return math.log(float(up)) if float(up) != 0 else 0
    else:
        return math.log(float(lp)) if float(lp) != 0 else 0


def parse_input_learning(filename : str, from_string : str = "") -> 'tuple[list[list[str]],list[list[str]],str,dict[str,float]]':
    '''
    #example(pos,Id,'atom') where Id is the Id of the (partial) answer set and atom is the correspondent atom
    #test(IdList)
    #train(IdList)
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
    # target = ""
    prob_facts_dict : dict[str,float] = dict()
    interpretations_dict : dict[int,list[str]] = dict()
    
    training_set : list[list[str]] = []
    test_set : list[list[str]] = []

    train_ids : list[int] = []
    test_ids : list[int] = []

    while i < len(lines):
        lines[i] = lines[i].replace('\n','')
        if lines[i].startswith("#program('"):
            i = i + 1
            while(not (lines[i].startswith("')."))):
                program = program + lines[i]
                i = i + 1
        # elif lines[i].startswith("#target("):
        #     ll = lines[i].split("#target(")
        #     target = ll[1].replace('\n','')[:-2]
        #     i = i + 1
        elif lines[i].startswith("#learnable("):
            ll = lines[i].split("#learnable(")
            name = ll[1].replace('\n','')[:-2]
            prob_facts_dict[name] = 0.5
            i = i + 1
        elif lines[i].startswith("#positive("):
            ll = lines[i].split("#positive(")
            id_interpretation = int(ll[1].split(',')[0])
            atom = ll[1].replace('\n','')[len(str(id_interpretation)) + 1 : -2]
            if id_interpretation in interpretations_dict.keys():
                interpretations_dict[id_interpretation].append(atom)
            else:
                interpretations_dict[id_interpretation] = [atom]
            i = i + 1
        elif lines[i].startswith("#negative("):
            ll = lines[i].split("#negative(")
            id_interpretation = int(ll[1].split(',')[0])
            atom = ll[1].replace('\n','')[len(str(id_interpretation)) + 1 : -2]
            if id_interpretation in interpretations_dict.keys():
                interpretations_dict[id_interpretation].append(f"not {atom}")
            else:
                interpretations_dict[id_interpretation] = [f"not {atom}"]

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

    for id in train_ids:
        training_set.append(interpretations_dict[int(id)])

    for id in test_ids:
        test_set.append(interpretations_dict[int(id)])

    # for el in interpretations_dict:
    #     print(f"{el}: {interpretations_dict[el]}")

    # print("Training set:")
    # print(training_set)
    # print("Test set:")
    # print(test_set)
    # print("Test ids:")
    # print(test_ids)
    # print("Train ids:")
    # print(train_ids)
    # print("Program:")
    # print(program)

    # import sys
    # sys.exit()

    return training_set, test_set, program, prob_facts_dict

def learn_parameters(
    training_set : 'list[list[str]]', 
    test_set : 'list[list[str]]', 
    program : str,
    prob_facts_dict : 'dict[str,float]',
    upper : bool = False, 
    verbose : bool = False) -> None:

    # start_time = time.time()

    # associate every interpretation i (int of the dict) to a list
    # that represents the id of the world as a 01 string and two integers
    # that indicates if contributes to the lower and upper probability
    # Example: the interpretation 1 has the world 0110 and 1010 that
    # contributes respectively to upper and both lower and upper
    # 1 -> [ [0110,0,1], [1010,1,1] ] 
    interpretations_to_worlds : 'dict[int,list[tuple[str,int,int]]]' = dict()

    ll0 = -10000000
    epsilon = 10e-5

    # compute negative LL
    p = 0
    for i in range(0, len(training_set)):
        lp, up = compute_probability_interpretation(
            prob_facts_dict, training_set[i], program, i, interpretations_to_worlds)
        p = p + to_logprob(lp, up, upper)

    ll1 = p

    # devo mantenere un dict che associa P(I) ad una lista di id di mondi
    # ed un dict che associa P(f_i | I) ad una lista di id di mondi

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
                # sum_expected_val_true = 0
                # sum_expected_val_false = 0
                # compute expected val
                s = generate_program_string(
                    prob_facts_dict, training_set[i], program)

                # Expectation: compute E[C_i1 | e]
                # e = target_predicate if pos_neg_training[i] > 0 else "np"

                pasta_solver = pasta.Pasta("", key, interpretation_string)
                lp, up = pasta_solver.inference(from_string=s)

                upper1 = upper1 + float(up)
                lower1 = lower1 + float(lp)

                # Expectation: compute E[C_i0 | e]
                pasta_solver = pasta.Pasta("", "nfp", interpretation_string)
                s = s + f"nfp:- not {key}.\n"
                lp, up = pasta_solver.inference(from_string=s)

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
            lp, up = compute_probability_interpretation(
                prob_facts_dict, training_set[i], program, i, interpretations_to_worlds)
            p = p + to_logprob(lp, up, upper)

        ll1 = p

    print(f"ll0: {ll0} ll1: {ll1} Iterations: {n_iterations}")
    print(prob_facts_dict)

    # end_time = time.time() - start_time

    # test_results(test_set, prob_facts_dict, program)


if __name__ == "__main__":

    # program = "background_example_bongard_dummy.lp"
    # program = "../../examples/learning/background_bayesian_network.lp"
    program = "../../examples/learning/background_shop.lp"
    # program = "../../examples/learning/background_smoke.lp"
    # program = "bongard_stress.lp"
    # program = "smoke_stress.lp"

    training_set, test_set, program, prob_facts_dict = parse_input_learning(program)
    upper = True
    verbose = False
    start_time = time.time()
    learn_parameters(training_set, test_set, program, prob_facts_dict, upper, verbose)
    end_time = time.time() - start_time

    print(f"Elapsed time: {end_time}")

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

