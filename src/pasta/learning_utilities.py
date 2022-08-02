import math

import pasta_solver

interpretation_string = "interpretation"
LOGZERO = 0.001


def generate_program_string(
    facts_prob: 'dict[str,float]',
    offset: int,
    atoms: 'list[str]',
    program: str
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

    return s + to_assert + "\n"



def get_prob_from_id(id: str, facts_prob: 'dict[str,float]') -> float:
    index = 0
    probability = 1
    for el in facts_prob:
        contribution = facts_prob[el] if id[index] == '1' else (
            1 - facts_prob[el])
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
        facts_prob: 'dict[str,float]',
        example: 'list[str]',
        program: str,
        key: int,
        interpretations_to_worlds: 'dict[int,list[tuple[str,int,int,int]]]',
        offset : int
) -> 'tuple[float,float]':
    '''
    Computation of the probability of an interpretation: P(I)
    '''

    if key not in interpretations_to_worlds:
        s = generate_program_string(facts_prob, offset, example, program)

        pasta_solver_ins = pasta_solver.Pasta(
            "", interpretation_string)  # type: ignore
        up: float = 0
        lp: float = 0
        lp, up = pasta_solver_ins.inference(from_string=s)  # type: ignore

        add_element_to_dict(
            pasta_solver_ins.interface.model_handler.worlds_dict, interpretations_to_worlds, key)
    else:
        lp, up = get_prob_from_dict(interpretations_to_worlds, facts_prob, key)

    return lp, up


def compute_expected_values(
        facts_prob: 'dict[str,float]',
        offset: int,
        atoms: 'list[str]',
        program: str,
        prob_fact: str,
        key: int,
        computed_expectation_dict: 'dict[str,list[tuple[str,int,int,int]]]'
) -> 'tuple[float,float,float,float]':

    idT = prob_fact + "_T" + str(key)
    idF = prob_fact + "_F" + str(key)

    if idT not in computed_expectation_dict:
        # call the solver
        s = generate_program_string(facts_prob, offset, atoms, program)

        # Expectation: compute E[f_i = True | I]
        pasta_solver_ins = pasta_solver.Pasta(
            "", prob_fact, interpretation_string)  # type: ignore
        lp1, up1 = pasta_solver_ins.inference(from_string=s)  # type: ignore

        # store the computed worlds
        add_element_to_dict(
            pasta_solver_ins.interface.model_handler.worlds_dict, computed_expectation_dict, idT)

        # Expectation: compute E[f_i = False | I]
        pasta_solver_ins = pasta_solver.Pasta(
            "", "nfp", interpretation_string)  # type: ignore
        s = s + f"nfp:- not {prob_fact}.\n"
        lp0, up0 = pasta_solver_ins.inference(from_string=s)  # type: ignore

        # store the computed worlds
        add_element_to_dict(
            pasta_solver_ins.interface.model_handler.worlds_dict, computed_expectation_dict, idF)
    else:
        # get prob from dict
        lp1: float = 0
        up1: float = 0
        lp0: float = 0
        up0: float = 0

        lp1, up1 = get_conditional_prob_from_dict(
            computed_expectation_dict, facts_prob, idT)
        lp0, up0 = get_conditional_prob_from_dict(
            computed_expectation_dict, facts_prob, idF)

    return lp1, up1, lp0, up0

# test


def test_results(
    test_set: 'list[list[str]]',
    interpretations_to_worlds: 'dict[int,list[tuple[str,int,int,int]]]',
    prob_facts_dict: 'dict[str,float]',
    program: str,
    offset : int
) -> None:

    p = 0

    for i in range(0, len(test_set)):
        lp, up = compute_probability_interpretation(
            prob_facts_dict, test_set[i], program, i, interpretations_to_worlds, offset)

        # lp, up = compute_probability_interpretation(
        #     prob_facts_dict, nl, program, i + 1000, interpretations_to_worlds)
        p = p + to_logprob(lp, up, False)

    print(f"LL: {p}")

    return None


def to_logprob(
    lp: float,
    up: float,
    upper: bool
) -> float:
    if upper:
        return math.log(float(up)) if float(up) != 0 else math.log(LOGZERO)
    else:
        return math.log(float(lp)) if float(lp) != 0 else math.log(LOGZERO)


def learn_parameters(
    training_set: 'list[list[str]]',
    test_set: 'list[list[str]]',
    program: str,
    prob_facts_dict: 'dict[str,float]',
    offset: int,
    upper: bool = False,
    verbose: bool = True
) -> 'dict[int,list[tuple[str,int,int,int]]]':

    # start_time = time.time()

    # associates every interpretation i (int of the dict) to a list
    # that represents the id of the world as a 01 string and three integers
    # that indicates the number of models for the not query, the number
    # of models for the query and the total number of models
    # Example: the interpretation 1 has the world 0110 and 1010 that
    # with certain values for the counts
    # 1 -> [ [0110,0,1,1], [1010,1,1,1] ]
    interpretations_to_worlds: 'dict[int,list[tuple[str,int,int,int]]]' = dict(
    )

    ll0 = -10000000
    epsilon = 10e-5

    computed_expectation_dict: 'dict[str,list[tuple[str,int,int,int]]]' = dict(
    )

    # compute negative LL
    p = 0
    for i in range(0, len(training_set)):
        lp, up = compute_probability_interpretation(
            prob_facts_dict, training_set[i], program, i, interpretations_to_worlds, offset)
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
        expected_dict: 'dict[str,list[float]]' = {}
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
                    s = expected_dict[k][3] + expected_dict[k][2]
                    prob_facts_dict[k] = expected_dict[k][3] / s if s > 0 else 0
                # per lower
                else:
                    s = expected_dict[k][0] + expected_dict[k][1]
                    prob_facts_dict[k] = expected_dict[k][1] / s if s > 0 else 0
            else:
                offset = offset - 1

        # Compute negative LL
        p = 0
        # for ex in examples:
        for i in range(0, len(training_set)):
            lp, up = compute_probability_interpretation(
                prob_facts_dict, training_set[i], program, i, interpretations_to_worlds, offset)
            p = p + to_logprob(lp, up, upper)

        ll1 = p
        offset = offset_value

    print(f"ll0: {ll0} ll1: {ll1}")
    print(f"Iterations: {n_iterations}")
    print(prob_facts_dict)

    return interpretations_to_worlds
