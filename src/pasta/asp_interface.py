from tabnanny import verbose
import clingo
from typing import Union
import time

# local
import models_handler

class Context:
    def id(self, x):
        return x
    def seq(self, x, y):
        return [x, y]

def on_model(m):
    print (m)

class AspInterface:
    '''
    Parameters:
        - content: list with the program
    '''

    def __init__(self, program_minimal_set: list, evidence: list, asp_program : list, probabilistic_facts : dict, n_abducibles : int, precision : int = 3, verbose : bool = False) -> None:
        self.cautious_consequences : list = []
        self.program_minimal_set : list = program_minimal_set
        self.asp_program : list = asp_program
        self.lower_probability_query : int = 0
        self.upper_probability_query : int = 0
        self.upper_probability_evidence : int = 0
        self.lower_probability_evidence : int = 0
        self.precision : int = precision
        self.evidence : str = evidence
        # self.probabilistic_facts = probabilistic_facts # unused
        self.n_prob_facts : int = len(probabilistic_facts)
        self.n_abducibles : int = n_abducibles
        self.constraint_times_list : list = []
        self.computed_models : int = 0
        self.grounding_time : int = 0
        self.n_worlds : int = 0
        self.world_analysis_time : int = 0
        self.computation_time : int = 0
        self.abductive_explanations : list = []
        self.abduction_time : int = 0
        self.verbose : bool = verbose

    def get_lower_probability_query(self) -> float:
        return float(self.lower_probability_query)
    
    def get_upper_probability_query(self) -> float:
        return float(self.upper_probability_query)

    def get_lower_probability_evidence(self) -> float:
        return float(self.lower_probability_evidence)

    def get_upper_probability_evidence(self) -> float:
        return float(self.upper_probability_evidence)

    '''
    Parameters:
        - None
    Return:
        - str
    Behavior:
        compute the minimal set of facts
        needed to make the query true. This operation is performed
        only if there is not evidence.
        Cautious consequences
        clingo <filename> -e cautious
    '''
    def get_minimal_set_facts(self) -> float:
        ctl = clingo.Control(["--enum-mode=cautious"])
        for clause in self.program_minimal_set:
            ctl.add('base',[],clause)

        ctl.ground([("base", [])])
        start_time = time.time()

        temp_cautious = []
        with ctl.solve(yield_=True) as handle:
            for m in handle:
                # i need only the last one
                temp_cautious = str(m).split(' ')
            handle.get()

        for el in temp_cautious:
            # if el != '' and (el.split(',')[-2] + ')' if el.count(',') > 0 else el.split('(')[0]) in self.probabilistic_facts:
            if el != '':
                self.cautious_consequences.append(el)

        # sys.exit()
        clingo_time = time.time() - start_time

        return clingo_time

    '''
    Parameters:
        - None
    Return:
        - int: number of computed models
        - float: grounding time
        - float: computing probability time
    Behavior:
        compute the lower and upper bound for the query
        clingo 0 <filename> --project
    '''
    def compute_probabilities(self) -> None:
        ctl = clingo.Control(["0","--project"])
        for clause in self.asp_program:
            ctl.add('base',[],clause)

        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                ctl.add('base',[],":- not " + c + '.')
        
        start_time = time.time()
        ctl.ground([("base", [])])
        self.grounding_time = time.time() - start_time

        start_time = time.time()
        model_handler = models_handler.ModelsHandler(self.precision, self.n_prob_facts, self.evidence)

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                model_handler.add_value(str(m))
                self.computed_models = self.computed_models + 1
            handle.get()
        self.computation_time = time.time() - start_time

        # print(model_handler) # prints the models in world format

        start_time = time.time()
        self.lower_probability_query, self.upper_probability_query = model_handler.compute_lower_upper_probability()

        self.n_worlds = model_handler.get_number_worlds()

        self.world_analysis_time = time.time() - start_time


    def abduction_iter(self, n_abd: int, previously_computed : list) -> Union[str, float]:
        ctl = clingo.Control(["0", "--project"])
        for clause in self.asp_program:
            ctl.add('base', [], clause)

        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                ctl.add('base', [], ":- not " + c + '.')

        if self.n_prob_facts is 0:
            ctl.add('base', [], ':- not q.')
        ctl.add('base', [], 'abd_facts_counter(C):- #count{X : abd_fact(X)} = C.')
        ctl.add('base', [], ':- abd_facts_counter(C), C != ' + str(n_abd) + '.')

        for el in previously_computed:
            ctl.add('base', [], ':- ' + el + ".")

        start_time = time.time()
        ctl.ground([("base", [])])
        self.grounding_time = time.time() - start_time

        computed_models = []

        with ctl.solve(yield_=True) as handle:
            for m in handle:
                # print(m)
                computed_models.append(str(m))
                # n_models = n_models + 1
            handle.get()

        computation_time = time.time() - start_time

        return computed_models, computation_time

    '''
    Abduction
    '''
    def abduction(self):
        result = []
        start_time = time.time()
        abducibles_list = []
        model_handler = models_handler.ModelsHandler(self.precision, self.n_prob_facts, None)

        for i in range(0, self.n_abducibles + 1):
            currently_computed, exec_time = self.abduction_iter(i, abducibles_list)
            if self.verbose:
                print("Models with " + str(i) + " abducibles: " + str(len(currently_computed)))
                print(currently_computed)

            # TODO: gestire len(currently_computed) > 0 and i == 0 (vero senza abducibili)

            if self.n_prob_facts == 0:
                # currently computed: list of computed models
                for i in range(0,len(currently_computed)):
                    currently_computed[i] = currently_computed[i].split(' ')
                    result.append(currently_computed[i])
                
                self.computed_models = self.computed_models + len(currently_computed)

                # do this only if deterministic abduction
                if len(currently_computed) > 0:
                    for cc in currently_computed:
                        for el in cc:
                            if el != "q" and not el.startswith('not_abd'):
                                abducibles_list.append(el)
            else:
                for el in currently_computed:
                    model_handler.add_model_abduction(str(el))

                self.lower_probability_query, self.upper_probability_query = model_handler.compute_lower_upper_probability()

            # keep the best model
            self.lower_probability_query, self.upper_probability_query = model_handler.keep_best_model()
            self.constraint_times_list.append(exec_time)

        for el in model_handler.abd_worlds_dict:
            result.append(el[1:].split(' '))

        self.abduction_time = time.time() - start_time
        self.abductive_explanations = result

    
    # prints the ASP program
    def print_asp_program(self) -> None:
        for el in self.asp_program:
            print(el)
        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                print(":- not " + c + '.')
