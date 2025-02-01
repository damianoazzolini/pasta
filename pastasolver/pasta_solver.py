""" Main module for the PASTA solver """

import argparse
import math
import statistics
import multiprocessing

from .pasta_parser import PastaParser
from .asp_interface import AspInterface
from .utils import *
from . import generator
# from . import learning_utilities
from .learning_utilities import ParameterLearner
from .arguments import parse_args_wrapper, Arguments


def check_lp_up(lp : float, up : float) -> None:
    '''
    Checks whether lp =< up
    '''
    if (lp > up) or (int(lp * 10e8) > 10e8) or (int(up * 10e8) > 10e8):
        s = f"Error in computing probabilities\nLower: {lp:.8f}\nUpper: {up:.8f}"
        print_error_and_exit(s)


class Pasta:
    '''
    Main class of the PASTA solver
    '''
    def __init__(self, args : Arguments) -> None:
        self.args = args
        
        self.interface : AspInterface
        self.parser : PastaParser


    def _get_program_str_for_aspmc(self, from_string : str = "") -> str:
        '''
        Returns the program string to use into the aspmc solver
        '''
        self.parser = PastaParser(self.args.filename, self.args.query, self.args.evidence)
        f = self.parser.get_file_handler(from_string)
        program_str = f.readlines()
        f.close()
        program_str.append(f"query({self.args.query}).")
        program_str = '\n'.join(program_str)
        return program_str


    def parameter_learning(self, from_string : str = "") -> None:
        '''
        Parameter learning
        '''
        self.parser = PastaParser(self.args.filename)
        self.parser.query = "placeholder"
        # self.parser.parse(from_string)
        # print(self.parser.get_asp_program())
        
        training_set, test_set, program, prob_facts_dict, offset = self.parser.parse_input_learning(from_string)
        # print(training_set, test_set, program, prob_facts_dict, offset)
        # sys.exit()
        parameter_learner = ParameterLearner(
            training_set=training_set,
            test_set=test_set,
            program=program,
            prob_facts_dict=prob_facts_dict,
            offset=offset,
            upper=not self.consider_lower_prob,
            aspmc=self.args.aspmc,
            verbose=self.args.verbosity
        )
        interpretations_to_worlds, learned_probs = parameter_learner.learn_parameters(
            
        )
        parameter_learner.test_results(
            interpretations_to_worlds,
            {}
        )
        self.parser.reconstruct_parameters(learned_probs)


    def test_unsat_xor(self, arguments: argparse.Namespace, from_string : str = "") -> 'tuple[float,float]':
        '''
        Unsat testing with XOR
        '''
        import clingo
        self.setup_interface(from_string)
        n = len(self.interface.prob_facts_dict)
        delta = arguments.delta # higher this value, less accurate will be the result
        alpha = arguments.alpha # < 0.0042 from the paper
        epsilon = 10e-5
        r = n/delta if n != delta else 1 + epsilon
        t = math.ceil(math.log(r)/alpha)
        m_list : 'list[float]' = []
        u_list : 'list[int]' = []
        
        # for c in self.parser.get_asp_program():
        #     print(c)
        print(f"{n*t} calls")
        for i in range(0, n + 1):  # or n+1?
            print(f"Iteration {i}")
            map_states: 'list[float]' = []
            ii: int = 1
            attempts = 0
            for _ in range(0,t + 1):
                # for _ in range(1, t + 1):
                # compute xor, loop until I get all the instances SAT
                # print('-- init ---')
                ctl = clingo.Control(["-Wnone"])
                for clause in self.parser.get_asp_program():
                    ctl.add('base', [], clause)
                    # print(clause)

                for _ in range(0, i):
                    current_constraint = generator.Generator.generate_xor_constraint(n)
                    ctl.add('base', [], current_constraint)
                    # print(current_constraint)
                ctl.ground([("base", [])])
                # with ctl.solve(yield_=True) as handle:  # type: ignore
                #     for m in handle:
                #         print(str(m))

                # print('--- END ---')

                res = str(ctl.solve())
                # print(res)
                if str(res) == "UNSAT":
                    attempts += 1
                    
            u_list.append(attempts) 
        
        print("Usat per iteration")
        print(u_list)
        l1 = []
        for el in u_list:
            l1.append(el/t)
        print(l1)
        import sys
        sys.exit()


    def approximate_solve_xor(self, arguments : argparse.Namespace, from_string : str = "") -> 'tuple[float,float]':
        '''
        Approximate inference (upper probability) using XOR constraints
        '''
        self.parser = PastaParser(self.filename, self.query, self.evidence, for_asp_solver=True)
        self.consider_lower_prob = False
        self.for_asp_solver = True

        map_program, n_vars = self.parser.inference_to_mpe(from_string)
        map_program = map_program + f":- not {self.query}.\n"

        # n = math.ceil(math.log2(2**n_vars)) # useless
        n = n_vars
        delta = arguments.delta # higher this value, less accurate will be the result
        alpha = arguments.alpha # < 0.0042 from the paper
        epsilon = 10e-5
        r = n/delta if n != delta else 1 + epsilon
        t = math.ceil(math.log(r)/alpha)
        m_list : 'list[float]' = []
        # maximum number of attempts for finding a program with a 
        # MAP state
        max_attempts : int = 200
        
        print(n,t,r,delta,alpha)
        
        unsat_count : 'list[int]' = []
        
        # t = 10

        # if self.verbose:
        print(f"Probability median of {t} values for each iteration")
        print(f"At least {n*t} MAP queries")

        for i in range(0, n): # or n+1?
            print(f"Iteration {i}")
            map_states : 'list[float]' = []
            ii : int = 1
            attempts = 0
            while ii < t + 1:
            # for _ in range(1, t + 1):
                # compute xor, loop until I get all the instances SAT
                current_program = map_program
                for _ in range(0, i):
                    current_constraint = generator.Generator.generate_xor_constraint(n_vars)
                    current_program = current_program + current_constraint + "\n"

                prob, s = self.upper_mpe_inference(current_program)
                if prob >= 0:
                    ii = ii + 1
                    map_states.append(prob)
                else:
                    attempts = attempts + 1
                    if attempts > max_attempts:
                        ii = ii + 1
                        attempts = 0
                        print_warning(f"Exceeded the max number of attempts ({max_attempts}) to find a consistent program.\nIteration (n): {i}, element (t): {ii}\nResults may be inaccurate.")
                        # print(current_program)
            unsat_count.append(attempts)
            # print(map_states)
            m_list.append(statistics.median(map_states))
        
        res_l = m_list[0]
        res_u = m_list[0]

        for i in range(0, len(m_list) - 1):
            res_l += m_list[i+1]*(2**i)
            res_u += m_list[i+1]*(2**(i+1))

        print(m_list)
        print(unsat_count)
        return res_l if res_l <= 1 else 1, res_u if res_u <= 1 else 1 


    def setup_sampling(self, from_string: str = "", keep_hybrid : bool = False) -> None:
        # TODO, REFACTOR: remove and use setup interface with approx True
        '''
        Setup the variables for sampling
        '''
        self.parser = PastaParser(self.filename, self.query, self.evidence)
        self.parser.parse(from_string, approximate_version=True, keep_hybrid=keep_hybrid)
        asp_program = self.parser.get_asp_program_approx()

        self.interface = AspInterface(
            self.parser.probabilistic_facts,
            asp_program,
            self.evidence,
            [],
            self.parser.abducibles,
            self.verbose,
            self.pedantic,
            self.samples,
            stop_if_inconsistent=self.stop_if_inconsistent,
            normalize_prob=self.normalize_prob,
            upper = not self.consider_lower_prob,
            continuous_facts=self.parser.continuous_facts if keep_hybrid else {}
        )


    def test_consistency(self, just_test : bool = False, from_string : str = "") -> None:
        '''
        Test the consistency of a program by sampling.
        '''
        self.setup_sampling(from_string)
        tested, inconsistent, iterations = self.interface.check_inconsistency_by_sampling(just_test)
        ratio = len(inconsistent) / 2**len(self.interface.prob_facts_dict)
        if ratio == 0:
            if len(tested) == 2**len(self.interface.prob_facts_dict):
                print("Program consistent")
            else:
                print(f"Tested {len(tested)} out of {2**len(self.interface.prob_facts_dict)} worlds ({(len(tested)/2**len(self.interface.prob_facts_dict))*100}%) in {iterations} iterations: probably consistent")
        else:
            print("Inconsistent program")
            print(f"Inconsistent worlds: {inconsistent}")
            print(f"Tested {len(tested)} out of {2**len(self.interface.prob_facts_dict)} worlds ({(len(tested)/2**len(self.interface.prob_facts_dict))*100}%) in {iterations} iterations")


    def approximate_solve(self, arguments : argparse.Namespace, from_string : str = "") -> 'tuple[float,float]':
        '''
        Inference through sampling
        '''
        self.setup_sampling(from_string, keep_hybrid=arguments.approximate_hybrid)

        if self.processes > 16:
            print_error_and_exit("Too many processes, max 16 for safety.")

        results : 'list[tuple[float,float]]' = []
        # set the number of samples per process
        self.interface.n_samples = int(self.samples / self.processes)

        if self.pedantic:
            print(f"Spawning {self.processes} processes")
        with multiprocessing.Pool(processes=self.processes) as pool:
            if self.evidence == "" and (arguments.rejection is False and arguments.mh is False and arguments.gibbs is False):
                for i in pool.imap_unordered(self.interface.sample_query, [1]*self.processes):
                    results.append(i)
                # i = self.interface.sample_query()
                # results.append(i)
            elif self.evidence != "":
                if arguments.rejection:
                    for i in pool.imap_unordered(self.interface.rejection_sampling, [1]*self.processes):
                        results.append(i)
                elif arguments.mh:
                    for i in pool.imap_unordered(self.interface.mh_sampling, [1]*self.processes):
                        results.append(i)
                elif arguments.gibbs:
                    for i in pool.imap_unordered(self.interface.gibbs_sampling, [arguments.block]*self.processes):
                        results.append(i)
                else:
                    print_error_and_exit("Specify a sampling method")
            else:
                print_error_and_exit("Missing evidence")

        if self.pedantic:
            print(f"Results: {results}")
        return statistics.mean([result[0] for result in results]), statistics.mean([result[1] for result in results])


    def setup_interface(self, from_string : str = "", approx : bool = False) -> None:
        '''
        Setup clingo interface
        '''
        self.parser = PastaParser(self.filename, self.query, self.evidence, self.for_asp_solver, self.naive_dt, self.lpmln)
        self.parser.parse(from_string, approx)

        if self.minimal is False:
            content_find_minimal_set = []
        else:
            content_find_minimal_set = self.parser.get_content_to_compute_minimal_set_facts()

        asp_program = self.parser.get_asp_program(self.lpmln)

        # if not self.consider_lower_prob and self.query != "":
        #     asp_program.append(f":- not {self.query}.")

        self.interface = AspInterface(
            self.parser.probabilistic_facts,
            asp_program,
            self.evidence,
            content_find_minimal_set,
            abducibles_list=self.parser.abducibles,
            verbose=self.verbose,
            pedantic=self.pedantic,
            stop_if_inconsistent=self.stop_if_inconsistent,
            normalize_prob=self.normalize_prob,
            xor=self.xor,
            decision_atoms_list=self.parser.decision_facts,
            utilities_dict=self.parser.fact_utility,
            upper=not self.consider_lower_prob,
            n_probabilistic_ics= self.parser.n_probabilistic_ics,
            k_credal = self.k_credal,
            # constraints=self.parser.constraints_list,
            # objective_function=self.parser.objective_function,
            optimizable_facts=self.parser.optimizable_facts,
            reducible_facts=self.parser.reducible_facts
        )

        if self.minimal:
            self.interface.compute_minimal_set_facts()

        if self.pedantic and self.minimal:
            print("--- Minimal set of probabilistic facts ---")
            print(self.interface.cautious_consequences)
            print("---")

        if self.pedantic:
            self.interface.print_asp_program()
            if self.minimal:
                print("--- Program to find minimal sets ---")
                print(*content_find_minimal_set, sep='\n')
                print("---")


    def decision_theory_approximate(self,
        from_string: str = "",
        samples : int = 1000,
        popsize : int = 50,
        mutation_probability : float = 0.05,
        iterations : int = 1000,
        to_maximize : str = "upper") -> 'tuple[list[float],list[str]]':
        '''
        Approximate solver for decision theory.
        '''
        # TODO: check the setup of the interface, that must be done
        # for approximate inference
        self.setup_interface(from_string, True)
        # self.setup_sampling(from_string)
        return self.interface.decision_theory_approximate(
            initial_population_size=popsize,
            mutation_probability=mutation_probability,
            samples_for_inference=samples,
            max_iterations_genetic=iterations,
            to_maximize=to_maximize
        )


    def decision_theory_naive(
            self,
            from_string: str = "",
            no_mix : bool = False,
            opt : bool = False,
            approximate : bool = False,
            samples : int = 1000
        ) -> 'tuple[tuple[float,float,float],list[str]]':
        '''
        Naive implementation of decision theory, i.e., by enumerating
        all the strategies and by picking the best one.
        '''
        self.setup_interface(from_string)
        if opt:
            return self.interface.decision_theory_opt(
                    approximate=approximate,
                    samples=samples
                )
        else:
            return self.interface.decision_theory_naive_method(no_mix)


    def decision_theory_improved(self, from_string: str = "") -> 'tuple[list[float],list[str]]':
        '''
        Decision theory solver by computing the projected
        solutions.
        '''
        self.setup_interface(from_string)
        return self.interface.decision_theory_project()


    def abduction(self,
        threshold : float = -1,
        only_smallest : bool = False,
        one_shot : bool = False,
        from_string: str = ""
        ) -> 'tuple[float,float,list[list[str]]]':
        '''
        Probabilistic and deterministic abduction
        '''
        self.setup_interface(from_string)
        self.interface.abduction(threshold=threshold,only_smallest_cardinality=only_smallest,one_shot=one_shot)
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query

        check_lp_up(lp, up)

        return lp, up, remove_dominated_explanations(self.interface.abductive_explanations)


    def approximate_abduction(
        self,
        threshold: float = -1,
        only_smallest: bool = False,
        from_string: str = "",
        samples : int = 1000,
        popsize : int = 50,
        mutation_probability : float = 0.05,
        iterations : int = 1000,
        target_probability : str = "lower"
        ) -> 'tuple[float,float,list[list[str]]]':
        '''
        Probabilistic and deterministic abduction
        '''
        self.setup_interface(from_string)
        self.interface.n_samples = samples
        self.interface.abduction_approximate(
            threshold=threshold,
            only_smallest_cardinality=only_smallest,
            initial_population_size=popsize,
            mutation_probability=mutation_probability,
            samples_for_inference=samples,
            max_iterations_genetic=iterations,
            target_probability=target_probability
        )
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query

        return lp, up, remove_dominated_explanations(self.interface.abductive_explanations)


    def inference(self, from_string : str = "") -> 'tuple[float,float]':
        '''
        Exact inference
        '''
        self.setup_interface(from_string)
        # self.interface.identify_useless_variables()
        self.interface.compute_probabilities()
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query

        check_lp_up(lp, up)

        return lp, up

    
    def _get_cnf_aspmc(self, from_string : str = ""):
        '''
        Gets the CNF representation of the program from aspmc.
        '''
        from aspmc.programs.smprogram import SMProblogProgram
        import aspmc.config as config

        config.config["knowledge_compiler"] = "c2d"
        # time aspmc -m smproblog b_10_ground.lp -c -k c2d

        program_files = []

        # cycle_breaking = "tp"

        # self.parser = PastaParser(self.filename, self.query, self.evidence)
        # f = self.parser.get_file_handler(from_string)
        # program_str = f.readlines()
        # f.close()
        # program_str.append(f"query({self.query}).")
        program_str = self._get_program_str_for_aspmc(from_string)

        program = SMProblogProgram(program_str, program_files)
        program._decomposeGraph()
        program.tpUnfold()
        program.td_guided_both_clark_completion(adaptive=False, latest=False)

        return program.get_cnf()
    
    
    def inference_aspmc(self, from_string : str = "") -> 'tuple[float,float]':
        '''
        Exact inference using aspmc as engine.
        # use the nse environment
        '''
        strategy = "flexible"
        preprocessing = False
        cnf = self._get_cnf_aspmc(from_string=from_string)
        results = cnf.evaluate(strategy = strategy, preprocessing = preprocessing)

        if len(results) > 0:
            # added
            lp_res = results[0][0]
            up_res = results[1][0]
            query = "q"
            print(f"Lower probability: {query}: {' '*max(1,(20 - len(query)))}{lp_res}")
            print(f"Upper probability: {query}: {' '*max(1,(20 - len(query)))}{up_res}")

        return lp_res, up_res

    
    def convert(self, from_string : str = "") -> None:
        '''
        Print the HPASP converted into PASP.
        '''
        self.setup_interface(from_string)
        for l in self.interface.asp_program:
            if not l.startswith('0{') and not l.startswith('#'):
                print(l)
        # print(*self.interface.asp_program, sep='\n')
        for el in self.interface.prob_facts_dict:
            print(f"{self.interface.prob_facts_dict[el]:f}::{el}.")


    def inference_lpmln(self, from_string : str = "") -> 'float':
        '''
        Inference under the LPMLN semantics
        '''
        self.setup_interface(from_string)
        self.interface.compute_probability_lpmln(self.query)

        return self.interface.lower_probability_query


    def map_inference(self, from_string : str = "") -> 'tuple[float,list[list[str]]]':
        '''
        Maximum a posteriori (MAP) inference: find the state (world)
        with maximum probability where the evidence holds.
        Most probable explanation (MPE) is MAP where no evidence is present
        i.e., find the world with highest probability where the query is true.
        '''
        self.setup_interface(from_string)
        if len(self.parser.map_id_list) == 0:
            print_error_and_exit("Specify at least one map fact.")
        if len(self.parser.map_id_list) == len(self.interface.prob_facts_dict) and not self.consider_lower_prob and not self.stop_if_inconsistent and not self.normalize_prob:
            print_warning("Brave (upper) MPE can be solved in a faster way using the --solver flag.")
        # self.consider_lower_prob = True
        self.interface.compute_probabilities()
        max_prob, map_state = self.interface.model_handler.get_map_solution(
            self.parser.map_id_list, self.consider_lower_prob)

        if self.normalize_prob and self.interface.normalizing_factor != 0:
            max_prob = max_prob / (1 - self.interface.normalizing_factor)

        return max_prob, map_state

    
    def reducible_task(
        self,
        target : str,
        threshold : float,
        simplify_iter : int = -1,
        from_string : str = ""
        ):
        '''
        Reducible task. Find the minimal subset of reducible facts
        such that the probability of the query is above the threshold.
        '''
        self.setup_interface(from_string)
        return self.interface.reducible_task(target, threshold, simplify_iter)


    def optimize_probability(
            self,
            target : str,
            threshold : float,
            epsilon : float,
            method : str,
            chunk : int = 100,
            credal_facts : bool = False,
            from_string : str = ""
        ):
        '''
        Optimize the probability of the optimizable facts subject to the constraints.
        Returns the result of scipy.minimize.
        '''
        self.setup_interface(from_string)

        return self.interface.optimize_prob(target, threshold, epsilon, method, chunk, credal_facts)


    def upper_mpe_inference(self, from_string : str = "") -> 'tuple[float,list[list[str]]]':
        '''
        MPE inference considering the upper probability.
        We suppose that every world has at least one answer set.
        '''
        self.setup_interface(from_string)
        if len(self.parser.map_id_list) == len(self.interface.prob_facts_dict):
            map_state, unsat = self.interface.compute_mpe_asp_solver(self.one)
            if unsat:
                probability = -1
                map_state_parsed = [["UNSAT"]]
            else:
                probability, map_state_parsed = self.interface.model_handler.extract_prob_from_map_state(map_state)
        else:
            print_error_and_exit("MAP inference cannot be solved with an ASP solver. Remove the --solver option.")
        
        return probability, map_state_parsed


def main():
    args = parse_args_wrapper()
    
    if args.profile:
        import cProfile, pstats, io
        from pstats import SortKey
        pr = cProfile.Profile()
        pr.enable()

    if args.query == "" and (not args.lpmln) and (args.test is None) and (args.uxor is None) and (args.dtn is None) and (args.dt is None):
        print_error_and_exit("Missing query")
    elif args.lpmln:
        if args.query == "" and not args.all:
            print_error_and_exit("Specify a query or use --all")
        if args.all:
            args.query = "__placeholder__"
    elif args.test is not None:
        args.query = "__placeholder__"
    elif args.convert:
        args.query = "asdf"
    if args.normalize and args.evidence:
        print_error_and_exit("Cannot use normalization and evidence.")
    if args.rejection or args.mh or args.gibbs:
        args.approximate = True
    if args.dtn and not args.approximate:
        print_warning("Naive decision theory solver, you should use -dt.")
    if args.map and args.solver:
        print_warning("Computing the upper MPE state, the program is assumed to be consistent.")
        args.upper = True
        args.minimal = False
        args.stop_if_inconsistent = False
        args.normalize = False
    if ((args.minimal and args.stop_if_inconsistent) or args.upper) and (not args.dtn and not args.dt and not args.dtopt and not args.map):
        print_warning("The program is assumed to be consistent.")
        args.stop_if_inconsistent = False
    if args.stop_if_inconsistent:
        args.minimal = False
    if args.pedantic:
        args.verbose = True
        
    if args.no_mix and (args.dt or args.dtn):
        print_warning("The lower utility may be greater than the upper utility for some strategies.")


    pasta_solver = Pasta(filename=args.filename,
                         query=args.query,
                         evidence=args.evidence,
                         verbose=args.verbose,
                         pedantic=args.pedantic,
                         samples=args.samples,
                         consider_lower_prob=not args.upper,
                         minimal=args.minimal,
                         normalize_prob=args.normalize,
                         stop_if_inconsistent=args.stop_if_inconsistent,
                         one=args.one,
                         xor=args.xor,
                         k=100,
                         naive_dt=args.dtn,
                         lpmln=args.lpmln,
                         processes=args.processes,
                         aspmc=args.aspmc
                        )

    if args.convert:
        pasta_solver.convert()
    elif args.abduction:
        if args.approximate:
            lower_p, upper_p, abd_explanations = pasta_solver.approximate_abduction(
                threshold=float(args.threshold),
                only_smallest=args.only_smallest,
                samples=args.samples
            )
        else:
            lower_p, upper_p, abd_explanations = pasta_solver.abduction(
                threshold=float(args.threshold),
                only_smallest=args.only_smallest,
                one_shot=args.one_shot
            )
        print_result_abduction(
            lower_p,
            upper_p,
            abd_explanations,
            args.upper,
            float(args.threshold),
            not args.only_smallest
        )
    elif args.xor:
        lower_p, upper_p = pasta_solver.approximate_solve_xor(args)
        print_prob(lower_p, upper_p)
    elif (args.approximate or args.approximate_hybrid) and not (args.dt or args.dtn or args.dtopt):
        lower_p, upper_p = pasta_solver.approximate_solve(args)
        print_prob(lower_p, upper_p)
    elif args.pl:
        pasta_solver.parameter_learning()
    elif args.map:
        if args.solver:
            pasta_solver.for_asp_solver = True
            max_p, atoms_list_res = pasta_solver.upper_mpe_inference()
        else:
            max_p, atoms_list_res = pasta_solver.map_inference()
        print_map_state(max_p, atoms_list_res, len(pasta_solver.interface.prob_facts_dict),args.upper)
    elif (args.dt or args.dtn) and args.approximate:
        if args.dt:
            print_error_and_exit("Approximate should be used with the -dtn flag.")
        best_util, utility_atoms = pasta_solver.decision_theory_approximate(
            samples=args.samples,
            popsize=args.popsize,
            mutation_probability=args.mutation,
            iterations=args.iterations)
        print(f"Utility: {best_util}\nChoice: {utility_atoms}")
    elif args.dtn or args.dtopt:
        lowest_val, lowest_comb, highest_val, highest_comb = pasta_solver.decision_theory_naive(
            no_mix=args.no_mix, opt=args.dtopt, approximate=args.approximate, samples=args.samples)
        print(f"Lowest utility: {lowest_val}\nChoice: {lowest_comb}")
        print(f"Highest utility: {highest_val}\nChoice: {highest_comb}")
    elif args.dt:
        if args.normalize:
            print_error_and_exit("Normalization should be used with the -dtn flag.")
        best_util, utility_atoms = pasta_solver.decision_theory_improved()
        print(f"Utility: {best_util}\nChoice: {utility_atoms}")
    elif args.test is not None:
        pasta_solver.test_consistency(args.test == 1)
    elif args.uxor:
        pasta_solver.test_unsat_xor(args)
    elif args.optimize or args.cf:
        res = pasta_solver.optimize_probability(
            target = args.target,
            threshold = args.threshold,
            epsilon = args.epsilon,
            method = args.method,
            chunk = args.chunk,
            credal_facts = args.cf
        )
        if not isinstance(res, list):
            res = [res]
        for idx, v in enumerate(res):
            if is_number(v):
                print(f"No optimization needed: {v}")
            else:
                if v.success:
                    if idx == 0:
                        print(f"Target function = {v.fun}")
                    else:
                        print(f"Target function = {str(v.fun)[1:]}")
                    print("Optimal probabilities")
                    print(v.x)
                else:
                    print_warning("Unable to solve the optimization problem.")
                if args.pedantic:
                    print(v)
    
    elif args.reducible:
        found, selected, computed_prob = pasta_solver.reducible_task(
            args.target,
            float(args.threshold)
        )
        if found:
            print("Solution found")
            for name, sel in selected.items():
                print(f"{name}: {sel}")
        else:
            print("Solution not found")
    else:
        if args.lpmln:
            prob = pasta_solver.inference_lpmln()
            lower_p = prob
            upper_p = prob
        elif args.aspmc:
            lower_p, upper_p = pasta_solver.inference_aspmc()
        else:
            lower_p, upper_p = pasta_solver.inference()
        if args.lpmln and args.all:
            for w in pasta_solver.interface.model_handler.worlds_dict:
                print(f"{w}: {pasta_solver.interface.model_handler.worlds_dict[w].prob}")
        else:
            print_prob(lower_p, upper_p, args.lpmln)

    if args.profile:
        pr.disable()
        s = io.StringIO()
        sortby = SortKey.CUMULATIVE
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())



if __name__ == "__main__":
    main()
