import math
import sys
import typing # for the Any type for the type hint
import random # to assign a random probability fo facts to construct the mapping for CNF
from . import pasta_solver

interpretation_string = "interpretation"
LOGZERO = 0.001


class ParameterLearner:
    def __init__(self,
            training_set: 'list[list[str]]',
            test_set: 'list[list[str]]',
            program: str,
            prob_facts_dict: 'dict[str,float]',
            offset: int,
            upper: bool = False,
            aspmc : bool = False,
            verbose: bool = True
        ) -> None:
        self.training_set = training_set
        self.test_set = test_set
        self.program = program
        self.prob_facts_dict = prob_facts_dict
        self.offset = offset
        self.upper = upper
        self.aspmc = aspmc
        self.verbose = verbose
        # offset is the index of the prob_facts_dict that separates
        # the prob facts with fixed probability from the ones that
        # should be learnt
        
        # maps the index of the cnf to the element is prob_facts_dict
        # I need one entry for every CNF
        self.cnf_idx_to_pf : 'dict[typing.Any,dict[int,str]]' = dict()


    def generate_program_string(
            self,
            atoms: 'list[str]'
        ) -> str:
        '''
        Generates a string containing the probabilistic answer set program.
        '''
        st_prog = ""
        st_prog = st_prog + self.program
        to_assert = f"{interpretation_string}:- {','.join(atoms)}.\n"
        offset = self.offset
        
        for k in self.prob_facts_dict:
            if offset == 0:
                st_prog = st_prog + f"{self.prob_facts_dict[k]}::{k}.\n"
            else:
                offset = offset - 1

        return st_prog + to_assert + "\n"



    def get_prob_from_id(self, id_w: str) -> float:
        '''
        Given a world id, extracts its probability
        '''
        probability = 1
        for index, el in enumerate(self.prob_facts_dict):
            contribution = self.prob_facts_dict[el] if id_w[index] == '1' else (1 - self.prob_facts_dict[el])
            probability = probability * contribution
        return probability


    def add_element_to_dict(self,
            worlds_dict, # type: ignore
            dict_to_store, # type: ignore
            key # type: ignore
        ):
        '''
        Adds an element to the specified dict.
        '''
        for w in worlds_dict: # type: ignore
            el = worlds_dict[w]  # type: ignore
            # if el.model_query_count != 0:
            if key not in dict_to_store:
                dict_to_store[key] = [[
                    w,
                    el.model_not_query_count, # type: ignore
                    el.model_query_count, # type: ignore
                    el.model_count # type: ignore
                ]]  # type: ignore
            else:
                dict_to_store[key].append([ # type: ignore
                    w, # type: ignore
                    el.model_not_query_count, # type: ignore
                    el.model_query_count, # type: ignore
                    el.model_count # type: ignore
                ])  # type: ignore


    def get_prob_from_dict(
            self,
            dict_with_data, # type: ignore
            key # type: ignore
        ) -> 'tuple[float,float]':
        '''
        Retrieves the probability of the query by looking into
        the already computed worlds.
        '''
        lp = 0
        up = 0
        worlds_list = dict_with_data[key] # type: ignore
        for world in worlds_list: # type: ignore
            w_id = world[0] # type: ignore
            mnqc = world[1] # type: ignore
            mqc = world[2] # type: ignore
            # mc = world[3]
            lpw = 1 if (mnqc == 0 and mqc > 0) else 0
            upw = 1 if mqc > 0 else 0

            current_prob = self.get_prob_from_id(w_id) # type: ignore

            lp = lp + current_prob * lpw
            up = up + current_prob * upw
        return lp, up


    def get_prob_from_dict_aspmc(
            self,
            aspmc_dict,
            key
        ) -> 'tuple[float,float]':
        '''
        Retrieves the probability of the query by looking into
        the already computed CNF for aspmc.
        '''
        current_cnf = aspmc_dict[key] # CNF representing the query
        print(self.cnf_idx_to_pf)
        print(current_cnf.quantified[0])
        for cnf_var in current_cnf.quantified[0]:
            idx_fact = self.cnf_idx_to_pf[str(hash(current_cnf))][cnf_var]
            current_cnf.weights[cnf_var][0] = self.prob_facts_dict[idx_fact]
            current_cnf.weights[-cnf_var][0] = 1-self.prob_facts_dict[idx_fact]

        results = current_cnf.evaluate(strategy="flexible", preprocessing=False)
        return results[0][0], results[1][0]


    def get_conditional_prob_from_dict(
            self,
            dict_with_data,
            key
        ) -> 'tuple[float,float]':
        '''
        Gets the conditional probability.
        '''

        # if key not in dict_with_data:
        #     return 0,0

        worlds_list = dict_with_data[key]

        lqp = 0
        uqp = 0
        lep = 0
        uep = 0

        for world in worlds_list:
            id_w = world[0]
            mnqe = world[1]
            mqe = world[2]
            nm = world[3]
            # print(world)
            # ['110', 0, 1]
            # [ID, Lower, Upper]
            # sys.exit()

            # print(facts_prob)
            current_prob = self.get_prob_from_id(id_w)

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

    
    def _compute_mappings_aspmc(self, cnf : 'typing.Any') -> None:
        '''
        Compute the mappings between CNF and probabilistic facts.
        Here I assume:
        - this is called only once at the beginning of the iterations, so:
        - it expects that all the learnable facts have probability 0.5
        '''

        for el in cnf.quantified[0]:
            # find the prob fact that has that weight
            for pf in self.prob_facts_dict:
                if self.prob_facts_dict[pf] == cnf.weights[el][0]:
                    if str(hash(cnf)) not in self.cnf_idx_to_pf:
                        self.cnf_idx_to_pf[str(hash(cnf))] = {}
                    self.cnf_idx_to_pf[str(hash(cnf))][el] = pf
            # restore the initial probability
            cnf.weights[el][0] = 0.5
            cnf.weights[-el][0] = 0.5


    def compute_probability_interpretation(
            self,
            example: 'list[str]',
            key: int,
            interpretations_to_worlds: 'dict[int,list[tuple[str,int,int,int]]]',
            aspmc_dict = {} # dict for aspmc that maps an index into a CNF
        ) -> 'tuple[float,float]':
        '''
        Computation of the probability of an interpretation: P(I)
        '''
        up: float = 0
        lp: float = 0
        
        if not self.aspmc:
            # PASTA
            if not (key in interpretations_to_worlds):
                s = self.generate_program_string(example)
                pasta_solver_ins = pasta_solver.Pasta("", interpretation_string)  # type: ignore
                lp, up = pasta_solver_ins.inference(from_string=s)  # type: ignore
                self.add_element_to_dict(pasta_solver_ins.interface.model_handler.worlds_dict, interpretations_to_worlds, key)
            else:
                lp, up = self.get_prob_from_dict(interpretations_to_worlds, key)
        else:
            # ASPMC
            if not (key in aspmc_dict):
                if len(self.cnf_idx_to_pf) < len(self.training_set):
                    # assign a random probability to every fact so I can recover them
                    for idx, pf in enumerate(self.prob_facts_dict):
                        if idx >= self.offset:
                            self.prob_facts_dict[pf] = random.random()
                            
                s = self.generate_program_string(example)
                pasta_solver_ins = pasta_solver.Pasta("", interpretation_string)  # type: ignore
                s = s.replace('__','')
                s = s.replace('not ','\+ ')
                cnf = pasta_solver_ins._get_cnf_aspmc(from_string=s)  # type: ignore
                # here I try to get the mappings between index and prob facts dict
                if len(self.cnf_idx_to_pf) < len(self.training_set):
                    # dict is empty: compute the mappings
                    self._compute_mappings_aspmc(cnf)
                    # print(self.cnf_idx_to_pf)
                    # sys.exit()
                    
                results = cnf.evaluate(strategy="flexible", preprocessing=False)

                lp = results[0][0]
                up = results[1][0]
                # up = pasta_solver_ins.inference_aspmc(from_string=s)
                aspmc_dict[key] = cnf
            else:
                lp, up = self.get_prob_from_dict_aspmc(
                    aspmc_dict,
                    key
                )
                print('found')
                print(lp,up)
                # sys.exit()

                
        return lp, up


    def compute_expected_values(
            self,
            offset: int,
            atoms: 'list[str]',
            prob_fact: str,
            key: int,
            computed_expectation_dict: 'dict[str,list[tuple[str,int,int,int]]]'
        ) -> 'tuple[float,float,float,float]':
        '''
        Computation of the expected values.
        '''

        idT = prob_fact + "_T" + str(key)
        idF = prob_fact + "_F" + str(key)

        if idT not in computed_expectation_dict:
            # call the solver
            s = self.generate_program_string(atoms)

            # Expectation: compute E[f_i = True | I]
            pasta_solver_ins = pasta_solver.Pasta(
                "", prob_fact, interpretation_string)  # type: ignore
            lp1, up1 = pasta_solver_ins.inference(from_string=s)  # type: ignore

            # store the computed worlds
            self.add_element_to_dict(
                pasta_solver_ins.interface.model_handler.worlds_dict, computed_expectation_dict, idT)

            # Expectation: compute E[f_i = False | I]
            pasta_solver_ins = pasta_solver.Pasta(
                "", "nfp", interpretation_string)  # type: ignore
            s = s + f"nfp:- not {prob_fact}.\n"
            lp0, up0 = pasta_solver_ins.inference(from_string=s)  # type: ignore

            # store the computed worlds
            self.add_element_to_dict(
                pasta_solver_ins.interface.model_handler.worlds_dict, computed_expectation_dict, idF)
        else:
            # get prob from dict
            lp1: float = 0
            up1: float = 0
            lp0: float = 0
            up0: float = 0

            lp1, up1 = self.get_conditional_prob_from_dict(computed_expectation_dict, idT)
            lp0, up0 = self.get_conditional_prob_from_dict(computed_expectation_dict, idF)

        return lp1, up1, lp0, up0

    # test


    def test_results(
        self,
        interpretations_to_worlds: 'dict[int,list[tuple[str,int,int,int]]]',
        aspmc_dict = {} # type: ignore
        ) -> None:
        '''
        Test the results.
        '''

        prob = 0

        for i, ith_test in enumerate(self.test_set):
            lp, up = self.compute_probability_interpretation( # type: ignore
                ith_test,
                i,
                interpretations_to_worlds,
                aspmc_dict # type: ignore
            )

            # lp, up = compute_probability_interpretation(
            #     prob_facts_dict, nl, program, i + 1000, interpretations_to_worlds)
            prob = prob + self.to_logprob(lp, up)

        print(f"LL: {prob}")


    def to_logprob(
        self,
        lp: float,
        up: float,
        ) -> float:
        '''
        Conversion to log probabilites.
        '''
        if self.upper:
            return math.log(float(up)) if float(up) != 0 else math.log(LOGZERO)
        else:
            return math.log(float(lp)) if float(lp) != 0 else math.log(LOGZERO)


    def learn_parameters(
        self,
        ) -> 'tuple[dict[int,list[tuple[str,int,int,int]]],dict[str,float]]':
        '''
        Main loop for parameter learning.
        '''

        # start_time = time.time()
        
        # if aspmc:
        #     import random
        #     # compute the CNF, extract the mapping between variable index in the 
        #     # prob dict and the index in the CNF
        #     map_dict : 'dict[str,float]' = {}
        #     for i, el in enumerate(prob_facts_dict):
        #         if i >= offset:
        #             map_dict[el] = random.random()
            
            

        # associates every interpretation i (int of the dict) to a list
        # that represents the id of the world as a 01 string and three integers
        # that indicates the number of models for the not query, the number
        # of models for the query and the total number of models
        # Example: the interpretation 1 has the world 0110 and 1010 that
        # with certain values for the counts
        # 1 -> [ [0110,0,1,1], [1010,1,1,1] ]
        interpretations_to_worlds: 'dict[int,list[tuple[str,int,int,int]]]' = dict()

        ll0 = -10000000
        epsilon = 10e-5

        computed_expectation_dict: 'dict[str,list[tuple[str,int,int,int]]]' = dict()
        
        aspmc_dict = dict()  # type: ignore

        # compute negative LL
        p = 0
        for i in range(0, len(self.training_set)):
            lp, up = self.compute_probability_interpretation( # type: ignore
                self.training_set[i],
                i,
                interpretations_to_worlds,
                aspmc_dict
            )
            p = p + self.to_logprob(lp, up)
            # print(f"interpretation: {i}")

        ll1 = p

        # FATTO: devo mantenere un dict che associa P(I) ad una lista di id di mondi
        # ed un dict che associa P(f_i | I) ad una lista di id di mondi

        # loop
        n_iterations = 0
        offset_value = self.offset
        offset = self.offset

        while (ll1 - ll0) > epsilon:
            n_iterations = n_iterations + 1
            print(f"ll0: {ll0} ll1: {ll1}")
            ll0 = ll1
            # fisso un fatto e calcolo la somma degli E per ogni esempio
            expected_dict: 'dict[str,list[float]]' = {}
            # Expectation
            for prob_fact in self.prob_facts_dict:
                if offset == 0:
                    upper0 = 0
                    upper1 = 0
                    lower0 = 0
                    lower1 = 0

                    for i in range(0, len(self.training_set)):
                        lp1, up1, lp0, up0 = self.compute_expected_values(
                            offset_value,
                            self.training_set[i],
                            prob_fact,
                            i,
                            computed_expectation_dict
                        )
                        upper1 = upper1 + up1
                        lower1 = lower1 + lp1
                        upper0 = upper0 + up0
                        lower0 = lower0 + lp0

                    expected_dict[prob_fact] = [lower0, lower1, upper0, upper1]
                else:
                    offset = offset - 1

            offset = offset_value

            if self.verbose:
                print(expected_dict)
                print(self.prob_facts_dict)

            # Maximization: update probabilities E[f_i = T | I] / (E[f_i = T | I] + E[f_i = F | I])
            for k in self.prob_facts_dict:
                if offset == 0:
                    # per upper
                    if self.upper:
                        s = expected_dict[k][3] + expected_dict[k][2]
                        self.prob_facts_dict[k] = expected_dict[k][3] / s if s > 0 else 0
                    # per lower
                    else:
                        s = expected_dict[k][0] + expected_dict[k][1]
                        self.prob_facts_dict[k] = expected_dict[k][1] / s if s > 0 else 0
                else:
                    offset = offset - 1

            # Compute negative LL
            p = 0
            # for ex in examples:
            for i in range(0, len(self.training_set)):
                lp, up = self.compute_probability_interpretation(  # type: ignore
                    self.training_set[i],
                    i,
                    interpretations_to_worlds,
                    aspmc_dict
                )
                p = p + self.to_logprob(lp, up)

            ll1 = p
            offset = offset_value

        print(f"ll0: {ll0} ll1: {ll1}")
        print(f"Iterations: {n_iterations}")
        print(self.prob_facts_dict)

        return interpretations_to_worlds, self.prob_facts_dict
