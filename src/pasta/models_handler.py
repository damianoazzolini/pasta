'''
Class to identify a world.
In the constructor, both lower and upper int (counters) are needed, since in 
some cases only model_not_query is true, so the world does not contribute
to the list
'''

from typing import Union

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
    
    def get_id(self) -> str:
        return self.id
    
    def get_prob(self) -> int:
        return self.prob

    def get_model_query_count(self) -> int:
        return self.model_query_count
    
    def get_model_not_query_count(self) -> int:
        return self.model_not_query_count
    
    def get_model_count(self) -> int:
        return self.model_count

    def increment_model_not_query_count(self) -> None:
        self.model_not_query_count = self.model_not_query_count + 1

    def increment_model_query_count(self) -> None:
        self.model_query_count = self.model_query_count + 1

    def increment_model_count(self) -> None:
        self.model_count = self.model_count + 1

    def __str__(self) -> str:
        return "id: " + self.id + " prob: " + str(self.prob) + \
            " mqc: " + str(self.get_model_query_count()) + \
            " mnqc: " + str(self.get_model_not_query_count()) + \
            " mc: " + str(self.get_model_count())
    
'''
Class to handle the models computed by clingo.
'''
class ModelsHandler():
    def __init__(self, precision : int, n_prob_facts : int, evidence : str) -> None:
        self.worlds_dict = dict()
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

    @staticmethod
    # ["bird(1,693)", "bird(2,693)", "bird(3,693)", "bird(4,693)", "nq", "ne"]
    # returns 11213141, 693 + 693 + 693 + 693, True
    # if q in line -> returns True else False in query
    # if e in line -> returns True else False in evidence
    # 11213141 means: 1 true, 2 true. 3 true, 4 true
    def get_id_prob_world(line: str, evidence: str) -> Union[str, int, bool, bool]:
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
                # TODO: here i suppose that everything is a compound with arity > 0, and not a term
                term = term.split('(')
                if term[1].count(',') == 0:  # arity original prob fact 0 (example: 0.2::a.)
                    id = id + term[0]
                    # if using log probabilities, replace * with +, also below
                    prob = prob * int(term[1][:-1])
                else:
                    args = term[1][:-1].split(',')
                    prob = prob * int(args[-1])
                    id = id + term[0]
                    for i in args[:-1]:
                        id = id + i

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

    # checks if the id is in the worlds list
    # query = True -> q in line
    # query = False -> nq in line
    # model_evidence = True -> e in line
    # model_evidence = False -> ne in line
    def manage_worlds_dict(self, id : str, prob : int, model_query : bool, model_evidence : bool) -> None:
        if id in self.worlds_dict:
            if self.evidence is None:
                if model_query == True:
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
        w = World(id,prob)
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
        # print(line)
        id, prob, model_query, model_evidence = self.get_id_prob_world(line,self.evidence)
        self.manage_worlds_dict(id, prob, model_query, model_evidence)
    
    # computes the lower and upper probability
    def compute_lower_upper_probability(self) -> Union[int,int,int,int]:
        for w in self.worlds_dict:
            p = (self.worlds_dict[w].get_prob() /
                 ((10**self.precision) ** self.n_prob_facts))
            if self.evidence is None:
                if self.worlds_dict[w].get_model_query_count() != 0:
                    if self.worlds_dict[w].get_model_not_query_count() == 0:
                        self.increment_lower_query_prob(p)
                    self.increment_upper_query_prob(p)
            else:
                mqe = self.worlds_dict[w].get_model_query_count()
                mnqe = self.worlds_dict[w].get_model_not_query_count()
                nm = self.worlds_dict[w].get_model_count()
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
        for el in self.worlds_dict:
            s = s + str(el) + "\n"
        return s
