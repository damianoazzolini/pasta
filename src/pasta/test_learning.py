import pasta_solver
import math
import time

import sys

import re

LOGZERO = 0.01

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
    offset : int,
    atoms : 'list[str]', 
    program : str
    ) -> str:
    '''
    Generates a string containing the PASP program
    '''
    
    s = ""
    s = s + program
    to_assert = f"{interpretation_string}:- "

    for e in atoms:
        to_assert = to_assert + e + ", "
    to_assert = to_assert[:-2] + ".\n"

    for k in facts_prob:
        if offset == 0:
            s = s + f"{facts_prob[k]}::{k}.\n"
        else:
            offset = offset - 1

    return s + to_assert +"\n"


# def get_tuple_regex(facts_prob : 'dict[str,float]'):
#     l = []
#     for el in facts_prob.keys():
#         e = el.replace('(','').replace(')','')
#         l.append(e)
#         l.append("not_" + e)
#     return tuple(l)


def get_prob_from_id(id : str, facts_prob : 'dict[str,float]') -> float:
    index = 0
    probability = 1
    for el in facts_prob:
        contribution = facts_prob[el] if id[index] == '1' else (1 - facts_prob[el])
        probability = probability * contribution
        index = index + 1
    return probability


def add_element_to_dict(worlds_dict, dict_to_store, key):
    for w in worlds_dict:
        el = worlds_dict[w]
        # if el.model_query_count != 0:
        if key not in dict_to_store: 
            dict_to_store[key] = [[
                w, 
                el.model_not_query_count,
                el.model_query_count,
                el.model_count
            ]]  # type: ignore
        else:
            dict_to_store[key].append([
                w, 
                el.model_not_query_count,
                el.model_query_count,
                el.model_count
            ])  # type: ignore


def get_prob_from_dict(dict_with_data, facts_prob, key):
    lp = 0
    up = 0
    worlds_list = dict_with_data[key]
    for world in worlds_list:
        id = world[0]
        mnqc = world[1]
        mqc = world[2]
        # mc = world[3]
        lpw = 1 if (mnqc == 0 and mqc > 0) else 0
        upw = 1 if mqc > 0 else 0
        # print(world)
        # ['110', 0, 1]
        # [ID, Lower, Upper]
        # sys.exit()

        # print(facts_prob)
        current_prob = get_prob_from_id(id, facts_prob)
        
        lp = lp + current_prob * lpw
        up = up + current_prob * upw
    return lp, up


def get_conditional_prob_from_dict(dict_with_data, facts_prob, key
    ) -> 'tuple[float,float]':

    # if key not in dict_with_data:
    #     return 0,0

    worlds_list = dict_with_data[key]

    lqp = 0
    uqp = 0
    lep = 0
    uep = 0

    for world in worlds_list:
        id = world[0]
        mnqe = world[1]
        mqe = world[2]
        nm = world[3]
        # print(world)
        # ['110', 0, 1]
        # [ID, Lower, Upper]
        # sys.exit()

        # print(facts_prob)
        current_prob = get_prob_from_id(id, facts_prob)

        if mqe > 0:
            if mqe == nm:
                lqp = lqp + current_prob
                # self.increment_lower_query_prob(p)
            uqp = uqp + current_prob
            # self.increment_upper_query_prob(p)
        if mnqe > 0:
            if mnqe == nm:
                lep = lep + current_prob
                # self.increment_lower_evidence_prob(p)
            uep = uep + current_prob
            # self.increment_upper_evidence_prob(p)

    if (uqp + lep == 0) and uep > 0:
        return 0, 0
    elif (lqp + uep == 0) and uqp > 0:
        return 1, 1
    else:
        if lqp + uep > 0:
            lqp = lqp / (lqp + uep)
        else:
            lqp = 0
        if uqp + lep > 0:
            uqp = uqp / (uqp + lep)
        else:
            uqp = 0
        return lqp, uqp


def compute_probability_interpretation(
        facts_prob : 'dict[str,float]',
        example : 'list[str]',
        program : str,
        key : int,
        interpretations_to_worlds: 'dict[int,list[tuple[str,int,int,int]]]'
        ) -> 'tuple[float,float]':
    '''
    Computation of the probability of an interpretation: P(I)
    '''

    if key not in interpretations_to_worlds:
        s = generate_program_string(facts_prob, offset, example, program)

        pasta_solver_ins = pasta_solver.Pasta("", interpretation_string)  # type: ignore
        up : float = 0
        lp : float = 0
        lp, up = pasta_solver_ins.inference(from_string = s)  # type: ignore

        add_element_to_dict(pasta_solver_ins.interface.model_handler.worlds_dict, interpretations_to_worlds, key)
    else:
        lp, up = get_prob_from_dict(interpretations_to_worlds, facts_prob, key)

    return lp, up
    

def compute_expected_values(
        facts_prob : 'dict[str,float]',
        offset : int,
        atoms : 'list[str]', 
        program : str,
        prob_fact : str,
        key : int,
        computed_expectation_dict : 'dict[str,list[tuple[str,int,int,int]]]'
        ) -> 'tuple[float,float,float,float]':

    idT = prob_fact + "_T" + str(key)
    idF = prob_fact + "_F" + str(key)

    if idT not in computed_expectation_dict:
        # call the solver
        s = generate_program_string(facts_prob, offset, atoms, program)

        # Expectation: compute E[f_i = True | I]
        pasta_solver_ins = pasta_solver.Pasta("", prob_fact, interpretation_string)  # type: ignore
        lp1, up1 = pasta_solver_ins.inference(from_string = s)  # type: ignore

        # store the computed worlds
        add_element_to_dict(pasta_solver_ins.interface.model_handler.worlds_dict, computed_expectation_dict, idT)

        # Expectation: compute E[f_i = False | I]
        pasta_solver_ins = pasta_solver.Pasta("", "nfp", interpretation_string)  # type: ignore
        s = s + f"nfp:- not {prob_fact}.\n"
        lp0, up0 = pasta_solver_ins.inference(from_string = s)  # type: ignore

        # store the computed worlds
        add_element_to_dict(pasta_solver_ins.interface.model_handler.worlds_dict, computed_expectation_dict, idF)
    else:
        # get prob from dict
        lp1 : float = 0
        up1 : float = 0
        lp0 : float = 0 
        up0 : float = 0

        lp1, up1 = get_conditional_prob_from_dict(computed_expectation_dict, facts_prob, idT)
        lp0, up0 = get_conditional_prob_from_dict(computed_expectation_dict, facts_prob, idF)

    return lp1, up1, lp0, up0

# test


def test_results(
    test_set : 'list[list[str]]',
    interpretations_to_worlds : 'dict[int,list[tuple[str,int,int,int]]]',
    prob_facts_dict: 'dict[str,float]',
    program: str
    ) -> None:

    # from sklearn.metrics import roc_auc_score
    # import numpy as np
    # from sklearn.metrics import (precision_recall_curve, PrecisionRecallDisplay)
    # import matplotlib.pyplot as plt
    # from sklearn.metrics import RocCurveDisplay

    # probs_lp : list[float] = []
    # probs_up : list[float] = []
    # ll = 0
    p = 0
    # pos_neg = []
    # ints = []
    for i in range(0, len(test_set)):
        # print(test_set[i])
        # pl = []
        # nl = []
        # for el in test_set[i]:
        #     if el.startswith('not '):
        #         nl.append(el.split('not ')[1])
        #     else:
        #         pl.append(el)
        # if len(nl) > 0:
        #     pos_neg.append(0)
        #     ints.append(nl)
        # if len(pl) > 0:
        #     pos_neg.append(1)
        #     ints.append(pl)

        lp, up = compute_probability_interpretation(
            prob_facts_dict, test_set[i], program, i, interpretations_to_worlds)

        # lp, up = compute_probability_interpretation(
        #     prob_facts_dict, nl, program, i + 1000, interpretations_to_worlds)
        p = p + to_logprob(lp, up, False)

        # if len(nl) > 0:
        #     probs_lp.append(lp)
        #     probs_up.append(up)
        
        # probs_lp.append(lp)
        # probs_up.append(up)

        # lp, up = compute_probability_interpretation(
        #     prob_facts_dict, pl, program, i - 1000, interpretations_to_worlds)

        # p = p + to_logprob(lp, up, False)
        
        # if len(pl) > 0:
        #     probs_lp.append(lp)
        #     probs_up.append(up)

        # nl = []
        # pl = []

    print(f"LL: {p}")

    # print(pos_neg)
    # print(ints)
    # print(probs_lp)

    # for i in range(0, len(test_set)):
    #     lp, up = compute_probability_interpretation(facts_prob, test_set[i], pos_neg_test_set[i], target_predicate, program, upper)
    #     probs_lp.append(lp)
    #     probs_up.append(up)
    #     ll = ll + to_logprob(lp, up, upper)
    
    # print(f"Negative LL: {ll}")
    # print(f"Probs lp: {probs_lp}")
    # print(f"Probs up: {probs_up}")
    # examples_0_1 = [1 for _ in range(0,len(test_set))]
    
    # import numpy as np
    # pos_neg_test_set_np = np.asarray(pos_neg, dtype=np.int0)
    # probs_lp_np = np.asarray(probs_lp, dtype=np.float32)
    # probs_up_np = np.asarray(probs_up, dtype=np.float32)

    # # print(pos_neg_test_set_np)
    # # print(probs_up_np)

    # auc_lp = roc_auc_score(pos_neg_test_set_np, probs_lp_np)
    # auc_up = roc_auc_score(pos_neg_test_set_np, probs_up_np)

    # print(f"auc_lp: {auc_lp}")
    # print(f"auc up: {auc_up}")
    
    # from sklearn import metrics
    # fpr, tpr, thresholds = metrics.roc_curve(pos_neg_test_set_np, probs_lp_np)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    # display.plot()
    # plt.show()

    # fpr, tpr, thresholds = metrics.roc_curve(pos_neg_test_set_np, probs_up_np)
    # roc_auc = metrics.auc(fpr, tpr)
    # display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,estimator_name='example estimator')
    # display.plot()
    # plt.show()

    # precision_lp, recall_lp, _ = precision_recall_curve(pos_neg_test_set_np, probs_lp_np)
    # disp = PrecisionRecallDisplay(precision=precision_lp, recall=recall_lp)
    # disp.plot()
    # plt.show()

    # precision_up, recall_up, _ = precision_recall_curve(pos_neg_test_set_np, )
    # disp = PrecisionRecallDisplay(precision=precision_up, recall=recall_up)
    # disp.plot()
    # plt.show()


    return None


def to_logprob(
    lp : float, 
    up : float, 
    upper : bool
    ) -> float:    
    if upper:
        return math.log(float(up)) if float(up) != 0 else math.log(LOGZERO)
    else:
        return math.log(float(lp)) if float(lp) != 0 else math.log(LOGZERO)


def parse_input_learning(filename : str, from_string : str = "") -> 'tuple[list[list[str]],list[list[str]],str,dict[str,float],int]':
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

    offset = 0

    while i < len(lines):
        lines[i] = lines[i].replace('\n','')
        if lines[i].startswith("#program('"):
            i = i + 1
            while(not (lines[i].startswith("')."))):
                program = program + lines[i]
                # look for prob facts in the program that need to be considered
                # in the dict but whose probabilities cannot be set
                if '::' in lines[i]:
                    prob_fact = lines[i].split('::')[1].replace('\n','').replace('.','').replace(' ','')
                    prob_facts_dict[prob_fact] = float(lines[i].split('::')[0])
                    offset = offset + 1
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

    return training_set, test_set, program, prob_facts_dict, offset

def learn_parameters(
    training_set : 'list[list[str]]', 
    test_set : 'list[list[str]]', 
    program : str,
    prob_facts_dict : 'dict[str,float]',
    offset : int,
    upper : bool = False, 
    verbose : bool = False
) -> 'dict[int,list[tuple[str,int,int,int]]]':

    # start_time = time.time()

    # associates every interpretation i (int of the dict) to a list
    # that represents the id of the world as a 01 string and three integers
    # that indicates the number of models for the not query, the number
    # of models for the query and the total number of models 
    # Example: the interpretation 1 has the world 0110 and 1010 that
    # with certain values for the counts
    # 1 -> [ [0110,0,1,1], [1010,1,1,1] ] 
    interpretations_to_worlds : 'dict[int,list[tuple[str,int,int,int]]]' = dict()

    ll0 = -10000000
    epsilon = 10e-5


    computed_expectation_dict : 'dict[str,list[tuple[str,int,int,int]]]' = dict()

    # compute negative LL
    p = 0
    for i in range(0, len(training_set)):
        lp, up = compute_probability_interpretation(
            prob_facts_dict, training_set[i], program, i, interpretations_to_worlds)
        p = p + to_logprob(lp, up, upper)
        # print(f"interpretation: {i}")

    ll1 = p

    # FATTO: devo mantenere un dict che associa P(I) ad una lista di id di mondi
    # ed un dict che associa P(f_i | I) ad una lista di id di mondi

    # loop
    n_iterations = 0
    offset_value = offset

    while (ll1 - ll0) > epsilon:
        n_iterations = n_iterations + 1
        print(f"ll0: {ll0} ll1: {ll1}")
        ll0 = ll1
        # fisso un fatto e calcolo la somma degli E per ogni esempio
        expected_dict : 'dict[str,list[float]]' = {}
        # Expectation
        for prob_fact in prob_facts_dict:
            if offset == 0:
                upper0 = 0
                upper1 = 0
                lower0 = 0
                lower1 = 0

                for i in range(0, len(training_set)):
                    lp1, up1, lp0, up0 = compute_expected_values(
                        prob_facts_dict, offset_value, training_set[i], program, prob_fact, i, computed_expectation_dict)

                    upper1 = upper1 + up1
                    lower1 = lower1 + lp1
                    upper0 = upper0 + up0
                    lower0 = lower0 + lp0

                expected_dict[prob_fact] = [lower0, lower1, upper0, upper1]
            else:
                offset = offset - 1

        offset = offset_value

        if verbose:
            print(expected_dict)
            print(prob_facts_dict)

        # Maximization: update probabilities E[f_i = T | I] / (E[f_i = T | I] + E[f_i = F | I])
        for k in prob_facts_dict:
            if offset == 0:
                # per upper
                if upper:
                    prob_facts_dict[k] = expected_dict[k][3] / \
                        (expected_dict[k][3] + expected_dict[k][2])
                # per lower
                else:
                    prob_facts_dict[k] = expected_dict[k][1] / \
                        (expected_dict[k][0] + expected_dict[k][1])
            else:
                offset = offset - 1

        # Compute negative LL
        p = 0
        # for ex in examples:
        for i in range(0, len(training_set)):
            lp, up = compute_probability_interpretation(
                prob_facts_dict, training_set[i], program, i, interpretations_to_worlds)
            p = p + to_logprob(lp, up, upper)

        ll1 = p
        offset = offset_value

    print(f"ll0: {ll0} ll1: {ll1}")
    print(f"Iterations: {n_iterations}")
    print(prob_facts_dict)

    return interpretations_to_worlds

    # end_time = time.time() - start_time

    # test_results(test_set, prob_facts_dict, program)


if __name__ == "__main__":

    # program = "background_example_bongard_dummy.lp"
    # program = "../../examples/learning/background_bayesian_network.lp"
    # program = "../../examples/learning/background_shop.lp"
    # program = "../../examples/learning/background_smoke.lp"
    # program = "bongard_stress.lp"
    # program = "smoke_stress.lp"
    program = "../../examples/learning/background_shop.lp"
    # program = "../../examples/learning/background_smoke_2.lp"

    training_set, test_set, program, prob_facts_dict, offset = parse_input_learning(program)
    upper = False
    verbose = False
    start_time = time.time()
    interpretations_to_worlds = learn_parameters(training_set, test_set, program, prob_facts_dict, offset, upper, verbose)
    end_time = time.time() - start_time

    print(f"Elapsed time: {end_time}")

    test_results(test_set,interpretations_to_worlds,prob_facts_dict,program)

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

