'''
Class to identify a world.
In the constructor, both lower and upper int (counters) are needed, since in 
some cases only model_not_query is true, so the world does not contribute
to the list
'''

from typing import Union

import sys

class AbdWorld:
    '''
    Class for the worlds defined by abducibles
    '''
    def __init__(self, id_abd : str,  id_prob : str, prob : int, model_query : bool) -> None:
        self.id : str = id_abd
        self.model_query_count : int = 0  # needed?
        self.model_not_query_count : int = 0  # needed?
        self.probabilistic_worlds : dict[str,World] = dict()

        if model_query is True:
            self.model_query_count = 1  # needed?
        else: 
            self.model_not_query_count = 1 # needed?
            
        self.probabilistic_worlds[id_prob] = World(id_prob, prob)
        if model_query is True:
            self.probabilistic_worlds[id_prob].increment_model_query_count()
        else:
            self.probabilistic_worlds[id_prob].increment_model_not_query_count()
    
    def manage_worlds_dict(self, id: str, prob: int, model_query: bool) -> None: 
        if model_query is True:
            self.model_query_count += 1  # needed?
        else: 
            self.model_not_query_count += 1 # needed?

        ModelsHandler.manage_worlds_dict(self.probabilistic_worlds, None, id, prob, model_query, None)

    def __str__(self) -> str:
        s = "id: " + self.id + " mqc: " + str(self.model_query_count) + \
            " mnqc: " + str(self.model_not_query_count) + "\n"
        
        for el in self.probabilistic_worlds:
            s = s + "\t" + self.probabilistic_worlds[el] + "\n"

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
        evidence : str
        ) -> None:
        self.worlds_dict : 'dict[str,World]' = dict()
        self.abd_worlds_dict = dict()
        self.prob_facts_dict = prob_facts_dict
        self.best_lp : int = 0 # best prob found so far with abduction
        self.best_up : int = 0 # best prob found so far with abduction
        self.best_abd_combinations = []
        self.upper_query_prob : float = 0
        self.lower_query_prob : float = 0
        self.upper_evidence_prob : float = 0
        self.lower_evidence_prob : float = 0
        self.n_prob_facts : int = len(prob_facts_dict)
        self.evidence : str = evidence


    def increment_lower_query_prob(self, p : float) -> None:
        self.lower_query_prob = self.lower_query_prob + p

    def increment_upper_query_prob(self, p : float) -> None:
        self.upper_query_prob = self.upper_query_prob + p

    def increment_lower_evidence_prob(self, p : float) -> None:
        self.lower_evidence_prob = self.lower_evidence_prob + p

    def increment_upper_evidence_prob(self, p: float) -> None:
        self.upper_evidence_prob = self.upper_evidence_prob + p

    def get_number_worlds(self) -> int:
        return len(self.worlds_dict.keys())

    def keep_best_model(self) -> Union[float,float]:
        # print(self.abd_worlds_dict)

        for el in self.abd_worlds_dict:
            acc_lp = 0
            acc_up = 0
            # print("id: " + el)
            worlds_comb = self.abd_worlds_dict[el].probabilistic_worlds
            for w_id in worlds_comb:
                # print(w_id)
                # print(worlds_comb[w_id].model_query_count)
                # print(worlds_comb[w_id].model_not_query_count)
                p = worlds_comb[w_id].prob
                if worlds_comb[w_id].model_query_count != 0:
                    acc_up = acc_up + p
                    if worlds_comb[w_id].model_not_query_count == 0:
                        acc_lp = acc_lp + p
            
            if acc_lp == self.best_lp and acc_lp > 0:
                # self.best_lp = acc_lp
                # self.best_up = acc_up
                self.best_abd_combinations.append(el)
            elif acc_lp > self.best_lp and acc_lp > 0:
                self.best_lp = acc_lp
                self.best_up = acc_up
                self.best_abd_combinations = []
                self.best_abd_combinations.append(el)

        # remove the dominated elements
        for el in list(self.abd_worlds_dict.keys()):
            if el not in self.best_abd_combinations:
                # print('removed: ' + el)
                del self.abd_worlds_dict[el]

        return self.best_lp, self.best_up

                    
    def extract_pos_and_prob(self, term : str) -> 'tuple[int,int,float]':
        '''
        Computes the position in the dict to generate the string and the probability
        of the current fact
        '''
        index = 0
        probability = 0
        positive = True
        if term.startswith('not_'):
            term = term.split('not_')[1]
            positive = False

        for el in self.prob_facts_dict:
            if term == el:
                probability = self.prob_facts_dict[el] if positive else 1 - \
                    self.prob_facts_dict[el]
                break
            else:
                index = index + 1
        
        return index, 1 if positive else 0, probability


    # def compute_world_probability(self, id : str) -> float:
    #     '''
    #     Computes the probability of a world in the form of 01 string
    #     '''
    #     return 1.0


    def get_id_prob_world(self, line: str, evidence: str) -> 'tuple[str, float, bool, bool]':
        '''
        From a line representing an answer set returns its id as a 01 string, its probability
        and whether it contributes to the lower and upper probability
        '''
        line_list = line.split(' ')
        model_query = False  # model q and e for evidence, q without evidence
        model_evidence = False  # model nq and e for evidence, nq without evidence
        id = "0" * len(self.prob_facts_dict)
        probability = 1
        for term in line_list:
            # print(f"Current term: {term}, len: {len(term)}")
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

        # print(f"id: {id}")
        # print(f"probability: {probability}")
        # print(f"model_query: {model_query}")
        if evidence == "" or evidence is None:
            # query without evidence
            return id, probability, model_query, False
        else:
            # can I return directly model_query and model_evidence?
            # also in the case of evidence == None
            if (model_query == True) and (model_evidence == True):
                return id, probability, True, True
            elif (model_query == False) and (model_evidence == True):
                return id, probability, False, True
            else:
                # all the other cases, don't care
                return id, probability, False, False

    # return: id_abd, id_prob, prob, model_query, similar to get_id_prob_world
    def get_ids_abduction(self, line : str) -> Union[str, str, int, bool]:
        # print(line)
        line = line.split(' ')
        model_query = False
        id_abd = ""
        id_prob = ""
        prob = 1
        for term in line:
            if term == "q":
                model_query = True
            elif term == "nq":
                model_query = False
            elif term.startswith('abd_') or term.startswith('not_abd_'):
                # abducible
                id_abd = id_abd + " " + term
            else:
                # probabilistic fact
                id_append, prob_mul = self.extract_id_append_and_prob(term)
                id_prob = id_prob + id_append
                # replace * with + for log probabilities
                prob = prob * prob_mul
        return id_abd, id_prob, prob, model_query


    def manage_worlds_dict(self, 
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
        if id in self.worlds_dict:
            if self.evidence == "" or self.evidence is None:
                if model_query is True:
                    self.worlds_dict[id].increment_model_query_count()
                else:
                    self.worlds_dict[id].increment_model_not_query_count()
            else:
                self.worlds_dict[id].increment_model_count()
                if (model_query == True) and (model_evidence == True):
                    self.worlds_dict[id].increment_model_query_count()  # q e
                elif (model_query == False) and (model_evidence == True):
                    self.worlds_dict[id].increment_model_not_query_count() # nq e
            return
        
        # element not found -> add a new world
        w = World(prob)
        if self.evidence is None:
            if model_query == True:
                w.increment_model_query_count()
            else:
                w.increment_model_not_query_count()
        else:
            w.increment_model_count()
            if (model_query == True) and (model_evidence == True):
                w.increment_model_query_count()  # q e
            elif (model_query == False) and (model_evidence == True):
                w.increment_model_not_query_count()  # nq e
        
        self.worlds_dict[id] = w


    # gets the stable model, extract the probabilities etc
    def add_value(self, line : str) -> None:
        '''
        Analyzes the stable models and construct the worlds
        '''
        # print(line)
        # print(self.prob_facts_dict)
        # sys.exit()
        id, probability, model_query, model_evidence = self.get_id_prob_world(line,self.evidence)
        # print(model_query)
        self.manage_worlds_dict(id, probability, model_query, model_evidence)

    def manage_worlds_dict_abduction(self, id_abd : str, id_prob : str, prob : int, model_query : bool) -> None:
        # check if the id of the abduction is present. If so, check for the
        # probability
        if id_abd in self.abd_worlds_dict:
            self.manage_worlds_dict(self.abd_worlds_dict[id_abd].probabilistic_worlds, None, id_prob, prob, model_query, None)
        else:
            # add new key
            self.abd_worlds_dict[id_abd] = AbdWorld(id_abd, id_prob, prob, model_query)

    # add_value for abduction
    def add_model_abduction(self, line : str) -> None:
        id_abd, id_prob, prob, model_query = self.get_ids_abduction(line)
        self.manage_worlds_dict_abduction(id_abd, id_prob, prob, model_query)


    def compute_lower_upper_probability(self) -> 'tuple[float,float]':
        '''
        Computes lower and upper probability
        '''
        for w in self.worlds_dict:
            p = self.worlds_dict[w].prob
            
            if self.evidence is None:
                if self.worlds_dict[w].model_query_count != 0:
                    if self.worlds_dict[w].model_not_query_count == 0:
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

        if self.evidence is None:
            return self.lower_query_prob, self.upper_query_prob
        else:
            if (self.upper_query_prob + self.lower_evidence_prob == 0) and self.upper_evidence_prob > 0:
                return 0,0
            elif (self.lower_query_prob + self.upper_evidence_prob == 0) and self.upper_query_prob > 0:
                return 1,1
            else:
                if self.lower_query_prob + self.upper_evidence_prob > 0:
                    lqp = self.lower_query_prob / (self.lower_query_prob + self.upper_evidence_prob)
                else:
                    lqp = 0
                if self.upper_query_prob + self.lower_evidence_prob > 0:
                    uqp = self.upper_query_prob / (self.upper_query_prob + self.lower_evidence_prob)
                else:
                    uqp = 0
                return lqp, uqp 


    def __repr__(self) -> str:
        s = ""
        if len(self.abd_worlds_dict) == 0:
            print("N worlds dict: " + str(len(self.worlds_dict)))
            for el in self.worlds_dict:
                s = s + str(el) + "\n"
        else:
            print("N abd worlds dict: " + str(len(self.abd_worlds_dict)))
            for el in self.abd_worlds_dict:
                s = s + str(el) + "\n"
        return s
