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
        self.model_query_count: int = 0  # needed?
        self.model_not_query_count: int = 0  # needed?
        self.probabilistic_worlds = dict()

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
            s = s + "\t" + self.probabilistic_worlds[el].__str__() + "\n"

        return s

class World:
    '''
    id is the string composed by the occurrences of the variables
    '''
    def __init__(self, id : str, prob : int) -> None:
        self.id : str = id
        self.prob : int = prob
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
        return "id: " + self.id + " prob: " + str(self.prob) + \
            " mqc: " + str(self.model_query_count) + \
            " mnqc: " + str(self.model_not_query_count) + \
            " mc: " + str(self.model_count)
    
'''
Class to handle the models computed by clingo.
'''
class ModelsHandler():
    def __init__(self, precision : int, n_prob_facts : int, evidence : str) -> None:
        self.worlds_dict = dict()
        self.abd_worlds_dict = dict()
        self.best_lp : int = 0 # best prob found so far with abduction
        self.best_up : int = 0 # best prob found so far with abduction
        self.best_abd_combinations = []
        self.upper_query_prob : float = 0
        self.lower_query_prob : float = 0
        self.upper_evidence_prob : float = 0
        self.lower_evidence_prob : float = 0
        self.precision : int = precision
        self.n_prob_facts : int = n_prob_facts
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

        for el in self.abd_worlds_dict:
            acc_lp = 0
            acc_up = 0
            # print("id: " + el)
            worlds_comb = self.abd_worlds_dict[el].probabilistic_worlds
            for w_id in worlds_comb:
                p = (worlds_comb[w_id].prob /
                     ((10**self.precision) ** self.n_prob_facts))
                if worlds_comb[w_id].model_query_count != 0:
                    acc_up = acc_up + p
                    if worlds_comb[w_id].model_not_query_count == 0:
                        acc_lp = acc_lp + p
            
            if acc_lp == self.best_lp:
                # self.best_lp = acc_lp
                # self.best_up = acc_up
                self.best_abd_combinations.append(el)
            elif acc_lp > self.best_lp:
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

                    
    def extract_id_append_and_prob(self, term : str) -> Union[str,int]:
        term = term.split('(')
        if term[1].count(',') == 0:  # arity original prob fact 0 (example: 0.2::a.)
            return term[0], int(term[1][:-1])
        else:
            args = term[1][:-1].split(',')
            prob = int(args[-1])
            id = ""
            id = id + term[0]
            for i in args[:-1]:
                id = id + i
            return id, prob

    # @staticmethod
    # ["bird(1,693)", "bird(2,693)", "bird(3,693)", "bird(4,693)", "nq", "ne"]
    # returns 11213141, 693 + 693 + 693 + 693, True
    # if q in line -> returns True else False in query
    # if e in line -> returns True else False in evidence
    # 11213141 means: 1 true, 2 true. 3 true, 4 true
    def get_id_prob_world(self, line: str, evidence: str) -> Union[str, int, bool, bool]:
        line = line.split(' ')
        model_query = False  # model q and e for evidence, q without evidence
        model_evidence = False  # model nq and e for evidence, nq without evidence
        id = ""
        prob = 1
        for term in line:
            if term == "q":
                model_query = True
            elif term == "nq":
                model_query = False
            elif term == "e":
                model_evidence = True
            elif term == "ne":
                model_evidence = False
            else:
                id_append, prob_mul = self.extract_id_append_and_prob(term)
                id = id + id_append
                # replace * with + for log probabilities
                prob = prob * prob_mul
                

        if evidence == None:
            # query without evidence
            return id, int(prob), model_query, False
        else:
            # can I return directly model_query and model_evidence?
            # also in the case of evidence == None
            if (model_query == True) and (model_evidence == True):
                return id, int(prob), True, True
            elif (model_query == False) and (model_evidence == True):
                return id, int(prob), False, True
            else:
                # all the other cases, don't care
                return id, int(prob), False, False

    # return: id_abd, id_prob, prob, model_query, similar to get_id_prob_world
    def get_ids_abduction(self, line : str) -> Union[str, str, int, bool]:
        # print(line)
        # sys.exit()
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
            elif '(' not in term:
                # abducible
                id_abd = id_abd + " " + term
            else:
                id_append, prob_mul = self.extract_id_append_and_prob(term)
                id_prob = id_prob + id_append
                # replace * with + for log probabilities
                prob = prob * prob_mul

        return id_abd, id_prob, prob, model_query

    # checks if the id is in the worlds list
    # query = True -> q in line
    # query = False -> nq in line
    # model_evidence = True -> e in line
    # model_evidence = False -> ne in line
    @staticmethod
    def manage_worlds_dict(worlds_dict : dict, evidence : str, id : str, prob : int, model_query : bool, model_evidence : bool) -> None:
        # print(worlds_dict)
        if id in worlds_dict:
            if evidence is None:
                if model_query is True:
                    worlds_dict[id].increment_model_query_count()
                else:
                    worlds_dict[id].increment_model_not_query_count()
            else:
                worlds_dict[id].increment_model_count()
                if (model_query == True) and (model_evidence == True):
                    worlds_dict[id].increment_model_query_count()  # q e
                elif (model_query == False) and (model_evidence == True):
                    worlds_dict[id].increment_model_not_query_count() # nq e
            return
        
        # element not found -> add a new world
        w = World(id,prob)
        if evidence is None:
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
        
        worlds_dict[id] = w

    # gets the stable model, extract the probabilities etc
    def add_value(self, line : str) -> None:
        # print(line)
        id, prob, model_query, model_evidence = self.get_id_prob_world(line,self.evidence)
        self.manage_worlds_dict(self.worlds_dict, self.evidence, id, prob, model_query, model_evidence)

    def manage_worlds_dict_abduction(self, id_abd : str, id_prob : str, prob : int, model_query : bool) -> None:
        # check if the id of the abduction is present. If so, check for the
        # probability
        if id_abd in self.abd_worlds_dict:
            # print('IN ' + id_abd)
            self.manage_worlds_dict(self.abd_worlds_dict[id_abd].probabilistic_worlds, None, id_prob, prob, model_query, None)
        else:
            # add new key
            # print('NEW ' + id_abd)
            self.abd_worlds_dict[id_abd] = AbdWorld(id_abd, id_prob, prob, model_query)

        # print("----------")
        # for el in self.abd_worlds_dict:
        #     print(self.abd_worlds_dict[el])

    # add_value for abduction
    def add_model_abduction(self, line : str) -> None:
        id_abd, id_prob, prob, model_query = self.get_ids_abduction(line)
        # print(id_abd)
        self.manage_worlds_dict_abduction(id_abd, id_prob, prob, model_query)

    # computes the lower and upper probability
    def compute_lower_upper_probability(self) -> Union[int,int]:
        for w in self.worlds_dict:
            p = (self.worlds_dict[w].prob /
                 ((10**self.precision) ** self.n_prob_facts))
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
            # print("upper evidence: " + str(self.upper_evidence_prob))
            # print("lower evidence: " + str(self.lower_evidence_prob))
            # print("upper query: " + str(self.upper_query_prob))
            # print("lower query: " + str(self.lower_query_prob))
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
                return lqp,uqp 


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
