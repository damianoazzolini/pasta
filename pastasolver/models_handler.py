'''
Class to identify a world.
'''

import math
from . import utils

class DecisionWorld:
    '''
    Class for storing the worlds defined by decision facts.
    '''
    def __init__(self,
        id_strategy : str,
        id_prob : str,
        prob : float,
        id_utilities : str
        ) -> None:
        self.id_strategy : str = id_strategy
        # each strategy has an associated set of worlds
        self.probabilistic_worlds : 'dict[str,World]' = {}
        # each world has an associated set of utility atoms true
        # or false, needed for the computation of the contribution
        # of the atoms
        self.probabilistic_worlds_to_utility : 'dict[str,list[str]]' = {}
        wrld = World(prob)
        self.probabilistic_worlds[id_prob] = wrld
        self.probabilistic_worlds_to_utility[id_prob] = [id_utilities]

    def __str__(self) -> str:
        str_decision = f"\nid decision: {self.id_strategy}\n"

        for world in self.probabilistic_worlds:
            str_decision += f"\tid world: {world}"
            str_decision += f" - probability: {self.probabilistic_worlds[world].prob}\n"
            str_decision += "\t\tutility atoms: "
            if world in self.probabilistic_worlds_to_utility:
                str_decision += f"{str(self.probabilistic_worlds_to_utility[world])}\n"
            else:
                str_decision += "\n"
            # s += f"\t\t{self.probabilistic_worlds_to_utility[self.probabilistic_worlds[world]]}\n"
        # for e in self.probabilistic_worlds_to_utility:
        #     s += f"\t\t{str(self.probabilistic_worlds_to_utility[e])}\n"

        return str_decision


    def __repr__(self) -> str:
        return self.__str__()


class AbdWorld:
    '''
    Class for the worlds defined by abducibles
    '''
    def __init__(self,
        id_abd : str,
        id_prob : str,
        prob : float,
        model_query : bool
        ) -> None:
        self.id_inst : str = id_abd
        self.model_query_count : int = 0  # needed?
        self.model_not_query_count : int = 0  # needed?
        self.probabilistic_worlds : 'dict[str,World]' = {}
        self.probabilistic_worlds[id_prob] = World(prob)
        if model_query is True:
            self.probabilistic_worlds[id_prob].increment_model_query_count()
        else:
            self.probabilistic_worlds[id_prob].increment_model_not_query_count()


    def __str__(self) -> str:
        s = "id: " + self.id_inst + " mqc: " + str(self.model_query_count) + \
            " mnqc: " + str(self.model_not_query_count) + "\n"

        for worlds in self.probabilistic_worlds.values():
            s = s + "\t" + worlds.__str__() + "\n"

        return s


    def __repr__(self) -> str:
        return self.__str__()


class World:
    '''
    id is the string composed by the occurrences of the variables
    '''
    def __init__(self, prob : float) -> None:
        # self.id : str = id
        self.prob: float = prob
        # meaning of these two:
        # if not evidence: model_not_query_count -> q
        #                  model_query_count -> q
        # if evidence: model_not_query_count -> q and e
        #              model_query_count -> nq and e
        self.model_not_query_count : int = 0
        self.model_query_count : int = 0
        # this is needed only on the case of evidence, to count the models
        self.model_count : int = 0

    def increment_model_not_query_count(self) -> None:
        '''
        Increments the number of not query count
        '''
        self.model_not_query_count = self.model_not_query_count + 1

    def increment_model_query_count(self) -> None:
        '''
        Increments the number of query count
        '''
        self.model_query_count = self.model_query_count + 1

    def increment_model_count(self) -> None:
        '''
        Increments the number stable model count
        '''
        self.model_count = self.model_count + 1

    def __str__(self) -> str:
        return "probability: " + str(self.prob) + \
            " mqc: " + str(self.model_query_count) + \
            " mnqc: " + str(self.model_not_query_count) + \
            " mc: " + str(self.model_count)

    def __repr__(self) -> str:
        return self.__str__()


class ModelsHandler():
    '''
    Class to handle the models computed by clingo
    '''
    def __init__(self,
        prob_facts_dict : 'dict[str,float]',
        evidence : str,
        abducibles_list : 'list[str]' = [],
        decision_atoms_list : 'list[str]' = [],
        utilities_dict : 'dict[str,float]' = {}
        ) -> None:
        self.worlds_dict : 'dict[str,World]' = {}
        self.abd_worlds_dict : 'dict[str,AbdWorld]' = {}
        self.prob_facts_dict = prob_facts_dict
        self.best_lp : float = 0
        self.best_up : float = 0
        self.best_abd_combinations : 'list[str]' = []
        self.upper_query_prob : float = 0
        self.lower_query_prob : float = 0
        self.upper_evidence_prob : float = 0
        self.lower_evidence_prob : float = 0
        self.evidence : str = evidence
        self.abducibles_list : 'list[str]' = abducibles_list # list of abducibles
        self.decision_atoms_list: 'list[str]' = decision_atoms_list
        self.utilities_dict: 'dict[str,float]' = utilities_dict
        self.decision_worlds_dict : 'dict[str,DecisionWorld]' = {}


    def keep_best_model(
        self,
        threshold : float = -1,
        normalize_prob: bool = False
        ) -> 'tuple[float,float]':
        '''
        Only keep the best model; used in abduction.
        If threshold > 0, constrained abduction: find the minimal set
        of facts such that the probability of the query is above the
        threshold.
        '''
        # current_number_abducibles = list(self.abd_worlds_dict.keys())[0].count('1')
        for el in self.abd_worlds_dict:
            acc_lp = 0
            acc_up = 0
            worlds_comb = self.abd_worlds_dict[el].probabilistic_worlds
            current_worlds = len(worlds_comb)
            expected_worlds = 0
            sum_p_worlds = 0
            world_prob = 0
            if current_worlds > 0:
                expected_worlds = 2**(len(list(worlds_comb.keys())[0]))
                # condition to maintain the consistency
            for w_id in worlds_comb:
                world_prob = worlds_comb[w_id].prob
                sum_p_worlds += world_prob
                if worlds_comb[w_id].model_query_count != 0:
                    acc_up = acc_up + world_prob
                    if worlds_comb[w_id].model_not_query_count == 0:
                        acc_lp = acc_lp + world_prob
            # print(acc_lp, acc_up, threshold)
            if current_worlds != expected_worlds:
                if normalize_prob:
                    acc_lp = acc_lp / world_prob
                    acc_up = acc_up / world_prob
                else:
                    # inconsistent, set everything to -1 to simulate
                    # the pruning
                    acc_lp = -1
                    acc_up = -1
                # consistent
            if threshold < 0:
                # plain abduction
                if acc_lp == self.best_lp and acc_lp > 0:
                    self.best_abd_combinations.append(el)
                elif acc_lp > self.best_lp and acc_lp > 0:
                    self.best_lp = acc_lp
                    self.best_up = acc_up
                    self.best_abd_combinations = []
                    self.best_abd_combinations.append(el)
            else:
                # constrained abduction
                # i don't need to separate the two cases
                # if acc_lp == threshold and acc_lp > 0:
                if acc_lp >= threshold and acc_lp > 0:
                    if el not in self.best_abd_combinations:
                        self.best_abd_combinations.append(el)
                    
                # elif acc_lp >= threshold and acc_lp > 0:
                #     self.best_abd_combinations = []
                #     self.best_abd_combinations.append(el)
                #     self.best_lp = acc_lp
                #     self.best_up = acc_up

        # remove the dominated elements
        for el in list(self.abd_worlds_dict.keys()):
            if el not in self.best_abd_combinations:
                del self.abd_worlds_dict[el]

        # print("To return")
        # print(self.best_abd_combinations)
        # print(self.best_lp, self.best_up)
        return self.best_lp, self.best_up


    def extract_pos_and_prob(self, term : str) -> 'tuple[int,int,float]':
        '''
        Computes the position in the dict to generate the string and the
        probability of the current fact
        '''
        index = 0
        probability = 0

        term, positive = utils.clean_term(term)

        found = False
        for el in self.prob_facts_dict:
            if term == el:
                probability = self.prob_facts_dict[el] if positive else 1 - self.prob_facts_dict[el]
                found = True
                break
            index = index + 1

        if found is False:
            utils.print_error_and_exit(f"Probabilistic fact {term} not found")

        return index, 1 if positive else 0, probability


    # this could be static or removed from the method
    def extract_pos(self, term : str, data_list : 'list[str]') -> 'tuple[int,int]':
        '''
        Computes the position in the list to get the index and the
        sign (positive or negative) for the current term.
        '''
        index = 0

        term, positive = utils.clean_term(term)

        for el in data_list:
            if term == el:
                break

            index = index + 1

        return index, 1 if positive else 0


    def get_id_prob_world(self,
        line: str,
        evidence: str
        ) -> 'tuple[str, float, bool, bool]':
        '''
        From a line representing an answer set returns its id as a 01 string, its probability
        and whether it contributes to the lower and upper probability
        '''
        line_list = line.split(' ')

        if len(line_list) < len(self.prob_facts_dict):
            # this because with the project statment the result will not
            # be correct: 0.5::a(1). a(X):- c(X). c(1). will provide a
            # wrong result
            utils.print_error_and_exit("Maybe a probabilistic fact has the same functor of a clause? Or you use not_f where f is a probabilistic fact.")

        model_query = False  # model q and e for evidence, q without evidence
        model_evidence = False  # model nq and e for evidence, nq without evidence
        id_str = "0" * len(self.prob_facts_dict)
        probability = 1
        for term in line_list:
            if term == "q":
                model_query = True
            elif term == "nq":
                model_query = False
            elif term == "e":
                model_evidence = True
            elif term == "ne":
                model_evidence = False
            else:
                position, true_or_false, prob = self.extract_pos_and_prob(term)
                id_str = id_str[:position] + str(true_or_false) + id_str[position + 1 :]
                probability = probability * prob

        if evidence == "":
            # query without evidence
            return id_str, probability, model_query, False

        # can I return directly model_query and model_evidence?
        # also in the case of evidence == ""?
        if (model_query is True) and (model_evidence is True):
            return id_str, probability, True, True
        if (model_query is False) and (model_evidence is True):
            return id_str, probability, False, True

        # all the other cases, don't care
        return id_str, probability, False, False


    def get_weight_as(self, line : str, query : str) -> 'tuple[float,bool]':
        '''
        Extracts the weight of a stable model
        '''
        l_splitted = line.split(' ')
        weight : float = 0.0

        for wr in self.prob_facts_dict:
            if wr in l_splitted:
                weight += weight + math.e**self.prob_facts_dict[wr]

        return weight if weight > 0 else 1, query in l_splitted


    def get_ids_abduction(self, line : str) -> 'tuple[str,str,float,bool]':
        '''
        From a line representing an answer set returns the id for both
        abducibles and worlds as a 01 string. Similar to get_id_prob_world
        '''
        line_list = line.split(' ')
        model_query = False
        id_abd = "0" * len(self.abducibles_list)
        id_prob = "0" * len(self.prob_facts_dict)

        probability = 1
        for term in line_list:
            if term == "q":
                model_query = True
            elif term == "nq":
                model_query = False
            elif term.startswith('abd_') or term.startswith('not_abd_'):
                position, true_or_false = self.extract_pos(term, self.abducibles_list)
                id_abd = id_abd[:position] + str(true_or_false) + id_abd[position + 1:]
            else:
                position, true_or_false, prob = self.extract_pos_and_prob(term)
                id_prob = id_prob[:position] + str(true_or_false) + id_prob[position + 1:]
                probability = probability * prob

        return id_abd, id_prob, probability, model_query


    def get_ids_decision(self, line: str) -> 'tuple[str,str,float,str]':
        '''
        From an answer set returns:
        id_strategy, id_world, prob_world, id_utilities
        '''
        line_list = line.split(' ')
        id_strategy = "0" * len(self.decision_atoms_list)
        id_world = "0" * len(self.prob_facts_dict)
        id_utilities = "0" * len(self.utilities_dict)
        prob_world = 1

        for term in line_list:
            t1, _ = utils.clean_term(term)
            if term.startswith("decision_"):
                position, true_or_false = self.extract_pos(term, self.decision_atoms_list)
                id_strategy = id_strategy[:position] + str(true_or_false) + id_strategy[position + 1:]
                # not very clean since clean_term is called both here and in extract_pos
            elif t1 in self.prob_facts_dict:
                position, true_or_false, prob = self.extract_pos_and_prob(term)
                id_world = id_world[:position] + str(true_or_false) + id_world[position + 1:]
                prob_world = prob_world * prob

            if t1 in self.utilities_dict:
                position, true_or_false = self.extract_pos(term, list(self.utilities_dict.keys()))
                id_utilities = id_utilities[:position] + str(true_or_false) + id_utilities[position + 1:]

        return id_strategy, id_world, prob_world, id_utilities


    def manage_worlds_dict(self,
        current_dict : 'dict[str,World]',
        id_w : str,
        prob : float,
        model_query : bool,
        model_evidence : bool
        ) -> None:
        '''
        Checks whether the id is in the list of worlds and update
        it accordingly.
        query = True -> q in line
        query = False -> nq in line
        model_evidence = True -> e in line
        model_evidence = False -> ne in line
        '''
        if id_w in current_dict:
            if self.evidence == "":
                if model_query is True:
                    current_dict[id_w].increment_model_query_count()
                else:
                    current_dict[id_w].increment_model_not_query_count()
                current_dict[id_w].increment_model_count()
            else:
                current_dict[id_w].increment_model_count()
                if (model_query is True) and (model_evidence is True):
                    current_dict[id_w].increment_model_query_count()  # q e
                elif (model_query is False) and (model_evidence is True):
                    current_dict[id_w].increment_model_not_query_count()  # nq e
            return

        # element not found -> add a new world
        w = World(prob)
        if self.evidence == "":
            if model_query is True:
                w.increment_model_query_count()
            else:
                w.increment_model_not_query_count()
            w.increment_model_count()
        else:
            w.increment_model_count()
            if (model_query is True) and (model_evidence is True):
                w.increment_model_query_count()  # q e
            elif (model_query is False) and (model_evidence is True):
                w.increment_model_not_query_count()  # nq e

        current_dict[id_w] = w


    def add_value(self, line : str) -> None:
        '''
        Analyzes the stable models and construct the world (credal semantics)
        '''
        w_id, probability, model_query, model_evidence = self.get_id_prob_world(line, self.evidence)
        self.manage_worlds_dict(self.worlds_dict, w_id, probability, model_query, model_evidence)


    def add_value_lpmln(self, line : str, query : str) -> float:
        '''
        Analyzes the answer set and store it, LPMLN semantics
        '''
        weight, model_query = self.get_weight_as(line, query)
        self.manage_worlds_dict(self.worlds_dict, line, weight, model_query, model_query)
        return weight


    def normalize_weights_as(self, nf : float) -> None:
        '''
        Normalizes the weights
        '''
        for el in self.worlds_dict:
            self.worlds_dict[el].prob = self.worlds_dict[el].prob/nf


    def manage_worlds_dict_abduction(self,
        id_abd : str,
        id_prob : str,
        prob : float,
        model_query : bool
        ) -> None:
        '''
        Checks whether the current id has been already encountered.
        If so, updates it; otherwise add a new element to the dict.
        '''
        if id_abd in self.abd_worlds_dict:
            # present
            self.manage_worlds_dict(self.abd_worlds_dict[id_abd].probabilistic_worlds, id_prob, prob, model_query, False)
        else:
            # add new key
            self.abd_worlds_dict[id_abd] = AbdWorld(id_abd, id_prob, prob, model_query)


    def manage_worlds_dict_decision(self,
        id_strategy: str,
        id_world: str,
        prob_world: float,
        id_utilities: str
        ) -> None:
        '''
        Checks whether the current id has been already encountered.
        If so, updates it; otherwise add a new element to the dict.
        '''
        if id_strategy in self.decision_worlds_dict:
            self.manage_worlds_dict(self.decision_worlds_dict[id_strategy].probabilistic_worlds, id_world, prob_world, True, False)
            if id_world in self.decision_worlds_dict[id_strategy].probabilistic_worlds_to_utility:
                self.decision_worlds_dict[id_strategy].probabilistic_worlds_to_utility[id_world].append(id_utilities)
            else:
                self.decision_worlds_dict[id_strategy].probabilistic_worlds_to_utility[id_world] = [id_utilities]
        else:
            self.decision_worlds_dict[id_strategy] = DecisionWorld(id_strategy, id_world, prob_world, id_utilities)


    def add_model_abduction(self, line : str) -> None:
        '''
        Adds a model for abductive reasoning
        '''
        id_abd, id_prob, prob, model_query = self.get_ids_abduction(line)
        self.manage_worlds_dict_abduction(id_abd, id_prob, prob, model_query)


    def add_decision_model(self, line : str) -> None:
        '''
        Adds a models for decision theory solving.
        Two possible options: aggregating the answer sets by worlds and,
        for each one, save which utilities are selected or viceversa.
        Here, the viceversa is used.
        '''
        id_strategy, id_world, prob_world, id_utilities = self.get_ids_decision(line)
        self.manage_worlds_dict_decision(id_strategy, id_world, prob_world, id_utilities)


    def compute_best_strategy(self, to_maximize : str = "upper") -> 'tuple[str,list[float]]':
        '''
        Computes the best strategy for decision theory.
        '''
        # utility_best_strategy : 'list[float]' = [-math.inf,-math.inf]
        decisions_utilities : 'dict[str,list[float]]' = {}
        best_strategy : str = ""
        bounds_best_strategy : 'list[float]' = [-math.inf, -math.inf]
        # print(self.decision_worlds_dict)

        for dw, el in self.decision_worlds_dict.items():
            # el = self.decision_worlds_dict[dw]
            lu_contr = 0
            uu_contr = 0
            for w in el.probabilistic_worlds:
                ual = el.probabilistic_worlds_to_utility[w]
                contribution = utils.sum_string_list(ual)
                l_utilities_dict = list(self.utilities_dict.values())
                current_world_prob = el.probabilistic_worlds[w].prob

                # print(f"world: {w} - probability: {current_world_prob}")
                for index, contr in enumerate(contribution):
                    if contr != 0:
                        current_reward = l_utilities_dict[index]
                        if current_reward > 0:
                            uu_contr += current_reward * current_world_prob
                            if contr == len(ual):
                                lu_contr += current_reward * current_world_prob
                        else:
                            utils.print_error_and_exit("Still to implement negative rewards with projected algorithm")

            decisions_utilities[dw] = [lu_contr,uu_contr]

        # print(decisions_utilities)
        for ut in decisions_utilities:
            if to_maximize == "lower":
                if bounds_best_strategy[0] < decisions_utilities[ut][0] or ((bounds_best_strategy[0] == decisions_utilities[ut][0]) and (bounds_best_strategy[1] < decisions_utilities[ut][1])):
                    best_strategy = ut
                    bounds_best_strategy = decisions_utilities[ut]
            elif to_maximize == "upper":
                if bounds_best_strategy[1] < decisions_utilities[ut][1] or ((bounds_best_strategy[1] == decisions_utilities[ut][1]) and (bounds_best_strategy[0] < decisions_utilities[ut][0])):
                    best_strategy = ut
                    bounds_best_strategy = decisions_utilities[ut]

        return best_strategy, bounds_best_strategy


    def get_abducibles_from_id(self, w_id : str) -> 'list[str]':
        '''
        From a 01 string returns the list of selected abducibles
        '''
        obtained_abds : 'list[str]' = []

        for i in range(0,len(w_id)):
            if w_id[i] == '1':
                obtained_abds.append(self.abducibles_list[i])
            # else:
            #     obtained_abds.append(f"not {self.abducibles_list[i]}")
        return obtained_abds


    def get_map_word_from_id(
        self,
        w_id : str,
        map_task : bool,
        map_id_list: 'list[int]'
        ) -> 'list[str]':
        '''
        From a 01 string returns the atoms in the world
        '''
        obtained_atoms : 'list[str]' = []
        ids_list : 'list[int]' = []
        keys = list(self.prob_facts_dict.keys())

        if map_task:
            ids_list = [i for i in range(0,len(self.prob_facts_dict))]
            for index, prob_fact in zip(ids_list, keys):
                if w_id[index] == '1':
                    obtained_atoms.append(prob_fact)
                else:
                    obtained_atoms.append(f"not {prob_fact}")
        else:
            for i, el in enumerate(map_id_list):
                if w_id[i] == '1':
                    obtained_atoms.append(keys[el])
                else:
                    obtained_atoms.append(f"not {keys[el]}")

        return obtained_atoms


    def compute_lower_upper_probability(self, k_credal : int = 100) -> 'tuple[float,float]':
        '''
        Computes lower and upper probability
        '''
        perc = k_credal / 100
        for w in self.worlds_dict:
            p = self.worlds_dict[w].prob

            if self.evidence == "":
                if self.worlds_dict[w].model_query_count != 0:
                    if int(perc) == 1:
                        if self.worlds_dict[w].model_not_query_count == 0:
                            self.lower_query_prob = self.lower_query_prob + p
                    else:
                        if self.worlds_dict[w].model_query_count/self.worlds_dict[w].model_count >= perc:
                            self.lower_query_prob = self.lower_query_prob + p
                    self.upper_query_prob = self.upper_query_prob + p
            else:
                mqe = self.worlds_dict[w].model_query_count
                mnqe = self.worlds_dict[w].model_not_query_count
                nm = self.worlds_dict[w].model_count
                if mqe > 0:
                    if mqe == nm:
                        self.lower_query_prob = self.lower_query_prob + p
                    self.upper_query_prob = self.upper_query_prob + p
                if mnqe > 0:
                    if mnqe == nm:
                        self.lower_evidence_prob = self.lower_evidence_prob + p
                    self.upper_evidence_prob = self.upper_evidence_prob + p

        if self.evidence == "":
            return self.lower_query_prob, self.upper_query_prob

        if (self.upper_query_prob + self.lower_evidence_prob == 0) and self.upper_evidence_prob > 0:
            return 0,0

        if (self.lower_query_prob + self.upper_evidence_prob == 0) and self.upper_query_prob > 0:
            return 1,1

        if self.lower_query_prob + self.upper_evidence_prob > 0:
            lqp = self.lower_query_prob / (self.lower_query_prob + self.upper_evidence_prob)
        else:
            lqp = 0

        if self.upper_query_prob + self.lower_evidence_prob > 0:
            uqp = self.upper_query_prob / (self.upper_query_prob + self.lower_evidence_prob)
        else:
            uqp = 0

        return lqp, uqp


    @staticmethod
    def get_sub_world(super_w : str, map_id_list : 'list[int]') -> str:
        '''
        Extracts a string from super_w representing a sub world.
        Example:
        super_w = 0101
        map_id_list = [0,2]
        result = 00 (extracts the values in position 0 and 2 of super_w)
        '''
        return ''.join([super_w[i] for i in map_id_list])


    def get_highest_prob_and_w_id_map(
        self,
        current_worlds_dict : 'dict[str,World]',
        map_id_list: 'list[int]',
        lower : bool = True,
        ) -> 'tuple[float,list[list[str]]]':
        '''
        Get the world with the highest associated probability
        '''
        max_prob : float = 0.0
        w_id_list : 'list[str]' = []
        
        print(current_worlds_dict)
        for el, w in current_worlds_dict.items():
            if (lower and w.model_query_count > 0 and w.model_not_query_count == 0) or (not lower and w.model_query_count > 0):
                # print("ok")
                if w.prob == max_prob:
                    max_prob = w.prob
                    w_id_list.append(el)
                elif w.prob > max_prob:
                    max_prob = w.prob
                    w_id_list = []
                    w_id_list.append(el)

        if max_prob == 0.0:
            return 0.0, []

        map_len = len(list(current_worlds_dict)[0]) == len(list(self.worlds_dict)[0])
        l_map_worlds = map(lambda w_id : self.get_map_word_from_id(w_id, map_len, map_id_list), w_id_list)
        return max_prob, list(l_map_worlds)


    def get_map_solution(
        self,
        map_id_list : 'list[int]',
        lower : bool = True
        ) -> 'tuple[float,list[list[str]]]':
        '''
        Analyzes the worlds obtained by the inference procedure and group
        them by map queries
        '''
        # pastasolver examples/map/simple_map_disj.pl --map --query="win": lower ed upper dovrebbero coincidere ma no
        if len(self.prob_facts_dict) == len(map_id_list):  # MPE: only map variables
            max_prob, atoms_list = self.get_highest_prob_and_w_id_map(self.worlds_dict, map_id_list, lower)
        else:
            # group by map variables
            # map_worlds : 'dict[str,World]' = {}
            # maps the map world to the lower and upper probability obtained by
            # the probabilistic worlds
            map_worlds_prob : 'dict[str,list[float]]' = {}
            for el, w in self.worlds_dict.items():
                if w.model_query_count > 0:
                    # keep both lower and upper
                    sub_w = ModelsHandler.get_sub_world(el, map_id_list)
                    if sub_w not in map_worlds_prob:
                        map_worlds_prob[sub_w] = [0,0]
                    if w.model_not_query_count == 0:
                        map_worlds_prob[sub_w][0] += w.prob
                    map_worlds_prob[sub_w][1] += w.prob # always increase the UP

            # get the sub-world with maximum probability
            max_prob : float = 0.0
            w_id_list : 'list[str]' = []
            target_pos = 0 if lower else 1
            
            for el, map_w in map_worlds_prob.items():
                if map_w[target_pos] == max_prob:
                    max_prob = map_w[target_pos]
                    w_id_list.append(el)
                elif map_w[target_pos] > max_prob:
                    max_prob = map_w[target_pos]
                    w_id_list = []
                    w_id_list.append(el)
            
            if max_prob == 0.0:
                return 0.0, []
            
            l_map_worlds = map(lambda w_id : self.get_map_word_from_id(w_id, False, map_id_list), w_id_list)
            atoms_list = list(l_map_worlds)
        
        return max_prob, atoms_list


    def extract_prob_from_map_state(self, map_state : str) -> 'tuple[float,list[list[str]]]':
        '''
        Extracts the probability form the MAP state computed with an
        ASP solver.
        '''
        probability : float = 1
        atoms : 'list[str]' = map_state.split(' ')
        map_state_parsed : 'list[str]' = []

        if atoms[0] == '':
            return 0, []

        for atm in atoms:
            if atm != 'q':
                atm = atm.split('not_')
                negated = len(atm) == 2
                atm_selected = atm[0] if len(atm) == 1 else atm[1]
                if '-' in atm_selected:
                    atm_selected = atm_selected.split('-')[0][:-1] + ')'
                current_p = self.prob_facts_dict[atm_selected] if not negated else (
                    1-self.prob_facts_dict[atm_selected])
                probability = probability * current_p

                map_state_parsed.append("not " + atm_selected if negated else atm_selected)

        # return [map_state_parsed] to have uniformity with MAP
        return probability, [map_state_parsed]


    def __repr__(self) -> str:
        str_repr = ""
        if len(self.abd_worlds_dict) == 0:
            print(f"N worlds dict: {len(self.worlds_dict)}")
            for wrld in self.worlds_dict:
                str_repr = str_repr + str(wrld) + "\n"
        else:
            print(f"N abd worlds dict: {len(self.abd_worlds_dict)}")
            for abd_wrld in self.abd_worlds_dict:
                str_repr = str_repr + str(abd_wrld) + "\n"
        return str_repr
