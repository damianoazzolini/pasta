'''
Class to identify a world.
'''

import utils
import math

class DecisionWorld:
    '''
    Class for storing the worlds defined by decision facts.
    '''
    def __init__(self,
        id_decision : str,
        id_prob : str,
        prob : float,
        id_utilities : str
        ) -> None:
        self.id_decision : str = id_decision
        self.probabilistic_worlds : 'dict[str,World]' = {}
        self.probabilistic_worlds_to_utility : 'dict[World,str]' = {}
        # mi serve anche una struttura che associa ad un mondo quali
        # utility atoms sono selezionati
        self.id_utilities : str = id_utilities
        self.probabilistic_worlds[id_prob] = World(prob)

    def __str__(self) -> str:
        s = f"id decision: {self.id_decision}, id utilities: {self.id_utilities}\n"

        for world in self.probabilistic_worlds:
            s = s + f"\t{world}\t{self.probabilistic_worlds[world].__str__()}\n"

        return s


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
        self.id : str = id_abd
        self.model_query_count : int = 0  # needed?
        self.model_not_query_count : int = 0  # needed?
        self.probabilistic_worlds : 'dict[str,World]' = {}
        self.probabilistic_worlds[id_prob] = World(prob)
        if model_query is True:
            self.probabilistic_worlds[id_prob].increment_model_query_count()
        else:
            self.probabilistic_worlds[id_prob].increment_model_not_query_count()


    def __str__(self) -> str:
        s = "id: " + self.id + " mqc: " + str(self.model_query_count) + \
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
        self.model_not_query_count = self.model_not_query_count + 1

    def increment_model_query_count(self) -> None:
        self.model_query_count = self.model_query_count + 1

    def increment_model_count(self) -> None:
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
        self.best_lp : float = 0 # best prob found so far with abduction
        self.best_up : float = 0 # best prob found so far with abduction
        self.best_abd_combinations : 'list[str]' = []
        self.upper_query_prob : float = 0
        self.lower_query_prob : float = 0
        self.upper_evidence_prob : float = 0
        self.lower_evidence_prob : float = 0
        self.n_prob_facts : int = len(prob_facts_dict)
        self.evidence : str = evidence
        self.abducibles_list : 'list[str]' = abducibles_list # list of abducibles
        self.decision_atoms_list: 'list[str]' = decision_atoms_list
        self.utilities_dict: 'dict[str,float]' = utilities_dict
        self.decision_worlds_dict : 'dict[str,DecisionWorld]' = {}


    def increment_lower_query_prob(self, p : float) -> None:
        self.lower_query_prob = self.lower_query_prob + p


    def increment_upper_query_prob(self, p : float) -> None:
        self.upper_query_prob = self.upper_query_prob + p


    def increment_lower_evidence_prob(self, p : float) -> None:
        self.lower_evidence_prob = self.lower_evidence_prob + p


    def increment_upper_evidence_prob(self, p: float) -> None:
        self.upper_evidence_prob = self.upper_evidence_prob + p


    def keep_best_model(self) -> 'tuple[float,float]':
        for el in self.abd_worlds_dict:
            acc_lp = 0
            acc_up = 0
            worlds_comb = self.abd_worlds_dict[el].probabilistic_worlds
            for w_id in worlds_comb:
                p = worlds_comb[w_id].prob
                if worlds_comb[w_id].model_query_count != 0:
                    acc_up = acc_up + p
                    if worlds_comb[w_id].model_not_query_count == 0:
                        acc_lp = acc_lp + p

            if acc_lp == self.best_lp and acc_lp > 0:
                self.best_abd_combinations.append(el)
            elif acc_lp > self.best_lp and acc_lp > 0:
                self.best_lp = acc_lp
                self.best_up = acc_up
                self.best_abd_combinations = []
                self.best_abd_combinations.append(el)

        # remove the dominated elements
        for el in list(self.abd_worlds_dict.keys()):
            if el not in self.best_abd_combinations:
                del self.abd_worlds_dict[el]

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
        if len(line_list) != len(self.prob_facts_dict) + 1:
            # this because with the project statment the result will not
            # be correct: 0.5::a(1). a(X):- c(X). c(1). will provide a 
            # wrong result
            utils.print_error_and_exit("Error: maybe a probabilitic facts has the same functor of a clause? Or you use not_f where f is a probabilistic fact.")

        model_query = False  # model q and e for evidence, q without evidence
        model_evidence = False  # model nq and e for evidence, nq without evidence
        id = "0" * len(self.prob_facts_dict)
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
                id = id[:position] + str(true_or_false) + id[position + 1 :]
                probability = probability * prob

        if evidence == "":
            # query without evidence
            return id, probability, model_query, False

        # can I return directly model_query and model_evidence?
        # also in the case of evidence == ""?
        if (model_query is True) and (model_evidence is True):
            return id, probability, True, True
        if (model_query is False) and (model_evidence is True):
            return id, probability, False, True

        # all the other cases, don't care
        return id, probability, False, False


    def get_weight_as(self, line : str, query : str) -> 'tuple[float,bool]':
        '''
        Extracts the weight of a stable model
        '''
        l = line.split(' ')
        weight : float = 0.0
        
        for wr in self.prob_facts_dict:
            if wr in l:
                weight += weight + math.e**self.prob_facts_dict[wr]

        return weight if weight > 0 else 1, query in l


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
        id_decision, id_world, prob_world, id_utilities
        '''
        line_list = line.split(' ')
        id_decision = "0" * len(self.decision_atoms_list)
        id_world = "0" * len(self.prob_facts_dict)
        id_utilities = "0" * len(self.utilities_dict)
        prob_world = 1

        for term in line_list:
            t1, _ = utils.clean_term(term)
            if term.startswith("decision_"):
                position, true_or_false = self.extract_pos(term, self.decision_atoms_list)
                id_decision = id_decision[:position] + str(true_or_false) + id_decision[position + 1:]
                # not very clean since clean_term is called both here and in extract_pos
            elif t1 in self.prob_facts_dict:
                position, true_or_false, prob = self.extract_pos_and_prob(term)
                id_world = id_world[:position] + str(true_or_false) + id_world[position + 1:]
                prob_world = prob_world * prob

            if t1 in self.utilities_dict:
                position, true_or_false = self.extract_pos(term, list(self.utilities_dict.keys()))
                id_utilities = id_utilities[:position] + str(true_or_false) + id_utilities[position + 1:]

        return id_decision, id_world, prob_world, id_utilities


    def manage_worlds_dict(self,
        current_dict : 'dict[str,World]',
        id : str,
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
        if id in current_dict:
            if self.evidence == "":
                if model_query is True:
                    current_dict[id].increment_model_query_count()
                else:
                    current_dict[id].increment_model_not_query_count()
                current_dict[id].increment_model_count()
            else:
                current_dict[id].increment_model_count()
                if (model_query is True) and (model_evidence is True):
                    current_dict[id].increment_model_query_count()  # q e
                elif (model_query is False) and (model_evidence is True):
                    current_dict[id].increment_model_not_query_count()  # nq e
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

        current_dict[id] = w


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
        id_decision: str,
        id_world: str,
        prob_world: float,
        id_utilities: str
        ) -> None:
        '''
        Checks whether the current id has been already encountered.
        If so, updates it; otherwise add a new element to the dict.
        '''
        if id_decision in self.decision_worlds_dict:
            self.manage_worlds_dict(self.decision_worlds_dict[id_decision].probabilistic_worlds, id_world, prob_world, True, False)
        else:
            self.decision_worlds_dict[id_decision] = DecisionWorld(id_decision,id_world,prob_world,id_utilities)


    def add_model_abduction(self, line : str) -> None:
        '''
        Adds a model for abductive reasoning
        '''
        id_abd, id_prob, prob, model_query = self.get_ids_abduction(line)
        self.manage_worlds_dict_abduction(id_abd, id_prob, prob, model_query)


    def add_decision_model(self, line : str) -> None:
        '''
        Adds a models for decision theory solving.
        Two possible options: aggregating the answer sets by worlds and, for each 
        one, save which utilities are selected or viceversa.
        Here, the viceversa is used.
        '''
        id_decision, id_world, prob_world, id_utilities = self.get_ids_decision(line)
        self.manage_worlds_dict_decision(id_decision, id_world, prob_world, id_utilities)
        # print(id_decision, id_world, prob_world, id_utilities)
        # import sys
        # sys.exit()
        # devo estrarre:
        # id_decision_atoms
        # id_world
        # probability_world
        # utility facts true e false (01 string?) e salvare per ognuno il conteggio (lower e upper probability)


    def compute_utility_atoms(self) -> None:
        print(self.decision_worlds_dict)
        import sys
        sys.exit()

    def get_abducibles_from_id(self, w_id : str) -> 'list[str]':
        '''
        From a 01 string returns the list of selected abducibles
        '''
        obtained_abds : 'list[str]' = []

        for i in range(0,len(w_id)):
            if w_id[i] == '1':
                obtained_abds.append(self.abducibles_list[i])
            else:
                obtained_abds.append(f"not {self.abducibles_list[i]}")
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
                            self.increment_lower_query_prob(p)
                    else:
                        if self.worlds_dict[w].model_query_count/self.worlds_dict[w].model_count >= perc:
                            self.increment_lower_query_prob(p)
                    self.increment_upper_query_prob(p)
            else:
                mqe = self.worlds_dict[w].model_query_count
                mnqe = self.worlds_dict[w].model_not_query_count
                nm = self.worlds_dict[w].model_count
                if mqe > 0:
                    if mqe == nm:
                        self.increment_lower_query_prob(p)
                    self.increment_upper_query_prob(p)
                if mnqe > 0:
                    if mnqe == nm:
                        self.increment_lower_evidence_prob(p)
                    self.increment_upper_evidence_prob(p)

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
        sub_w : str = ""
        for el in map_id_list:
            sub_w = sub_w + super_w[el]

        return sub_w


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

        for el in current_worlds_dict:
            w = current_worlds_dict[el]
            if w.model_query_count > 0 and (w.model_not_query_count == 0 if lower else True):
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
        if len(self.prob_facts_dict) == len(map_id_list):  # MPE: only map variables
            max_prob, atoms_list = self.get_highest_prob_and_w_id_map(self.worlds_dict, map_id_list, lower)
        else:
            # group by map variables
            map_worlds : 'dict[str,World]' = {}
            for el in self.worlds_dict:
                w = self.worlds_dict[el]
                if w.model_query_count > 0:
                    # keep both lower and upper
                    sub_w = ModelsHandler.get_sub_world(el, map_id_list)
                    if sub_w in map_worlds:
                        map_worlds[sub_w].model_query_count = map_worlds[sub_w].model_query_count + w.model_query_count
                        map_worlds[sub_w].model_not_query_count = map_worlds[sub_w].model_not_query_count + w.model_not_query_count
                        map_worlds[sub_w].prob = map_worlds[sub_w].prob + w.prob  # add the probability
                    else:
                        map_worlds[sub_w] = World(w.prob)
                        map_worlds[sub_w].model_query_count = map_worlds[sub_w].model_query_count + w.model_query_count
                        map_worlds[sub_w].model_not_query_count = map_worlds[sub_w].model_not_query_count + w.model_not_query_count

            # get the sub-world with maximum probability
            max_prob, atoms_list = self.get_highest_prob_and_w_id_map(map_worlds, map_id_list, lower)

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

        for a in atoms:
            if a != 'q':
                a = a.split('not_')
                negated = len(a) == 2
                a1 = a[0] if len(a) == 1 else a[1]
                if '-' in a1:
                    a1 = a1.split('-')[0][:-1] + ')'
                p = self.prob_facts_dict[a1] if not negated else (1-self.prob_facts_dict[a1])
                probability = probability * p
                
                map_state_parsed.append("not " + a1 if negated else a1)

        # return [map_state_parsed] to have uniformity with MAP
        return probability, [map_state_parsed]


    def __repr__(self) -> str:
        s = ""
        if len(self.abd_worlds_dict) == 0:
            print(f"N worlds dict: {len(self.worlds_dict)}")
            for el in self.worlds_dict:
                s = s + str(el) + "\n"
        else:
            print(f"N abd worlds dict: {len(self.abd_worlds_dict)}")
            for el in self.abd_worlds_dict:
                s = s + str(el) + "\n"
        return s
