'''
Class to identify a world.
In the constructor, both lower and upper int (counters) are needed, since in 
some cases only model_not_query is true, so the world does not contribute
to the list
'''

from typing import Union
from utilities import get_id_prob_world


class World():
    '''
    id is the string composed by the occurrences of the variables
    For example, for
    model_not_query(1,693,1,693,2,1832,0,0)
    the id is 1120
    '''
    def __init__(self, id : str, prob : int) -> None:
        self.id = id
        self.prob = prob
        self.lower = 0 # here i just need a Boolean
        self.upper = 0
    
    def get_id(self) -> str:
        return self.id
    
    def increment_lower(self) -> str:
        self.lower = self.lower + 1

    def increment_upper(self) -> str:
        self.lower = self.upper + 1

    # used to keep the list of worlds sorted
    def __eq__(self, o: object) -> bool:
        return self.id == o.id
    
    def __lt__(self, o: object) -> bool:
        return self.id < o.id
    
'''
Class to handle the models computed by clingo.
Each model is a line of the form
bird_a(2) bird_b(1) bird_b(2) nobird_a(1) model_not_query(1,693,1,693,2,1832,0,0)
'''
class ModelsHandler():
    def __init__(self) -> None:
        self.worlds_list = []

    # checks if the id is in the worlds list
    # query = True -> model_query
    # query = False -> model_not_query
    def manage_worlds_list(self, id : str, prob : int, query : bool) -> None:
        for el in self.worlds_list:
            if el.get_id() == id:
                if query == True:
                    el.increment_upper()
                else:
                    el.increment_lower()
                return
        
        # element not found -> add a new world
        w = World(id,prob)
        if query == True:
            w.increment_upper()
        else:
            w.increment_lower()

        self.worlds_list.append(w)
        self.worlds_list = sorted(self.worlds_list) # keep the list sorted

    # gets the stable model, extract the probabilities etc
    def add_value(self, line : str) -> None:
        line = line.split(' ')
        to_analyse = None
        for el in line:
            if "model_query" or "model_not_query" in el:
                to_analyse = el
                break
        
        if to_analyse is None:
            print("Error in analysing the models: no models found")

        el = to_analyse.split('(')[1][:-1] # extract the values
        id, prob = get_id_prob_world(el)
        if "model_query" in to_analyse:
            self.manage_worlds_list(id,prob,True)
        else: # model no query
            self.manage_worlds_list(id,prob,False)
    
    # computes the lower and upper probability
    def compute_lower_upper_probability(self) -> Union[float,float]:
        # TODO
        return 0,0