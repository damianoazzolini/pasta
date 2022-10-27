""" Main module for the PASTA solver """

import argparse
import math
import statistics

# from pasta.pasta_parser import PastaParser
from pasta_parser import PastaParser
# import pasta_parser
from asp_interface import AspInterface
# import asp_interface
from utils import print_error_and_exit, print_waring

import generator

import learning_utilities

examples_string_exact = "python3 pasta_solver.py \
    ../examples/bird_4.lp \
    --query=\"fly(1)\""
examples_string_exact_evidence = "python3 pasta_solver.py \
    ../examples/bird_4.lp \
    --query=\"fly(1)\" \
    --evidence=\"bird(1)\""
examples_string_approximate = "python3 pasta_solver.py \
    ../examples/bird_4.lp \
    --query=\"fly(1)\" \
    --approximate"
examples_string_approximate_rej = "python3 pasta_solver.py \
    ../examples/bird_4.lp \
    --query=\"fly(1)\" \
    --evidence=\"bird(1)\" --rejection"
examples_strings = "Examples:\n\n" + examples_string_exact + "\n\n" + examples_string_exact_evidence + \
    "\n\n" + examples_string_approximate + "\n\n" + examples_string_approximate_rej

pasta_description = "PASTA: Probabilistic Answer Set programming for STAtistical probabilities"


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
    def __init__(
        self,
        filename : str,
        query : str,
        evidence : str = "",
        verbose : bool = False,
        pedantic : bool = False,
        samples : int = 1000,
        consider_lower_prob : bool = True,
        minimal : bool = False,
        normalize_prob : bool = False,
        stop_if_inconsistent : bool = False,
        one : bool = False,
        xor : bool = False,
        k : int = 100
        ) -> None:
        self.filename = filename
        self.query = query
        self.evidence = evidence
        self.verbose = verbose
        self.pedantic = pedantic
        if pedantic is True:
            self.verbose = True
        self.samples = samples
        # lower or upper probability bound for MAP/Abduction, default lower
        self.consider_lower_prob = consider_lower_prob
        self.minimal = minimal
        self.normalize_prob = normalize_prob
        self.stop_if_inconsistent = stop_if_inconsistent
        self.for_asp_solver = False
        self.one = one
        self.xor = xor
        self.k_credal : int = k
        self.interface : AspInterface
        self.parser : PastaParser


    def parameter_learning(self, from_string : str = "") -> None:
        '''
        Parameter learning
        '''
        self.parser = PastaParser(self.filename)
        training_set, test_set, program, prob_facts_dict, offset = self.parser.parse_input_learning(from_string)
        interpretations_to_worlds = learning_utilities.learn_parameters(
            training_set, test_set, program, prob_facts_dict, offset, not self.consider_lower_prob, self.verbose)
        learning_utilities.test_results(test_set, interpretations_to_worlds, prob_facts_dict, program, offset)


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
        # print(n)
        t = math.ceil(math.log(n/delta)/alpha)
        m_list : 'list[float]' = []
        # maximum number of attempts for finding a program with a 
        # MAP state
        max_attempts : int = 200

        # if self.verbose:
        print(f"Probability median of {t} values for each iteration")
        print(f"At least {n*t} MAP queries")

        for i in range(0, n+1): # or n+1?
            print(f"Iteration {i}")
            map_states : 'list[float]' = []
            ii : int = 1
            attempts = 0
            while ii < t + 1:
            # for _ in range(1, t + 1):
                # compute xor, loop until I get all the instances SAT
                xor_constraints : 'list[str]' = []
                current_program = map_program
                for _ in range(0, i):
                    current_constraint = generator.Generator.generate_xor_constraint(n_vars)
                    xor_constraints.append(current_constraint)
                    current_program = current_program + current_constraint + "\n"
                prob, s = self.upper_mpe_inference(current_program)
                # print(xor_constraints)
                # print(s)
                if prob >= 0:
                    ii = ii + 1
                    map_states.append(prob)
                else:
                    attempts = attempts + 1
                    if attempts > max_attempts:
                        ii = ii + 1
                        attempts = 0
                        print_waring(f"Exceeded the max number of attempts to find a consistent program.\nIteration (n): {i}, element (t): {ii}\nResults may be inaccurate.")
                        # print(current_program)
            # print(map_states)
            m_list.append(statistics.median(map_states))
        
        res_l = m_list[0]
        res_u = m_list[0]

        for i in range(0, len(m_list) - 1):
            res_l += m_list[i+1]*(2**i)
            res_u += m_list[i+1]*(2**(i+1))

        return res_l if res_l <= 1 else 1, res_u if res_u <= 1 else 1 



    def approximate_solve(self, arguments : argparse.Namespace, from_string : str = "") -> 'tuple[float,float]':
        '''
        Inference through sampling
        '''
        self.parser = PastaParser(self.filename, self.query, self.evidence)
        self.parser.parse_approx(from_string)
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
            continuous_vars=self.parser.continuous_vars,
            constraints_list=self.parser.constraints_list,
            upper = not self.consider_lower_prob
        )

        if self.evidence == "" and (arguments.rejection is False and arguments.mh is False and arguments.gibbs is False):
            lp, up = self.interface.sample_query()
        elif self.evidence != "":
            if arguments.rejection is True:
                lp, up = self.interface.rejection_sampling()
            elif arguments.mh is True:
                lp, up = self.interface.mh_sampling()
            elif arguments.gibbs is True:
                lp, up = self.interface.gibbs_sampling(arguments.block)
            else:
                lp = 0
                up = 0
                print_error_and_exit("Specify a sampling method: Gibbs, MH, or Rejection.")
        else:
            print_error_and_exit("Missing evidence")

        return lp, up


    def setup_interface(self, from_string : str = "") -> None:
        '''
        Setup clingo interface
        '''
        self.parser = PastaParser(self.filename, self.query, self.evidence, self.for_asp_solver)
        self.parser.parse(from_string)

        if self.verbose:
            print("Parsed program")

        if self.minimal is False:
            content_find_minimal_set = []
        else:
            content_find_minimal_set = self.parser.get_content_to_compute_minimal_set_facts()

        asp_program = self.parser.get_asp_program()

        if not self.consider_lower_prob:
            asp_program.append(f":- not {self.query}.")

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
            n_probabilistic_ics= self.parser.n_probabilistic_ics
        )

        exec_time = 0
        if self.minimal:
            exec_time = self.interface.get_minimal_set_facts()

        if self.verbose:
            print(f"Computed cautious consequences in {exec_time} seconds")
            if self.pedantic and self.minimal:
                print("--- Minimal set of probabilistic facts ---")
                print(self.interface.cautious_consequences)
                print("---")

        if self.pedantic:
            print("--- Asp program ---")
            self.interface.print_asp_program()
            print("---")
            if self.minimal:
                print("--- Program to find minimal sets ---")
                for e in content_find_minimal_set:
                    print(e)
                print("---")


    def decision_theory(self, from_string: str = "") -> 'tuple[float,float,list[list[str]]]':
        self.setup_interface(from_string)
        self.interface.print_asp_program()
        print(self.interface.decision_atoms_list)
        print(self.interface.utilities_dict)
        self.interface.decision_theory()
        import sys
        sys.exit()


    def abduction(self, from_string: str = "") -> 'tuple[float,float,list[list[str]]]':
        '''
        Probabilistic and deterministic abduction
        '''
        self.setup_interface(from_string)
        self.interface.abduction()
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query

        check_lp_up(lp, up)

        return lp, up, self.interface.abductive_explanations


    def inference(self, from_string : str = "") -> 'tuple[float,float]':
        '''
        Exact inference
        '''
        self.setup_interface(from_string)
        # self.interface.identify_useless_variables()
        self.interface.compute_probabilities()
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query
        if self.interface.normalizing_factor >= 1:
            lp = 1
            up = 1
            print_waring("No worlds have > 1 answer sets")

        if self.normalize_prob and self.interface.normalizing_factor != 0:
            lp = lp / (1 - self.interface.normalizing_factor)
            up = up / (1 - self.interface.normalizing_factor)

        check_lp_up(lp, up)

        return lp, up


    def map_inference(self, from_string : str = "") -> 'tuple[float,list[list[str]]]':
        '''
        Maximum a posteriori (MAP) inference: find the state (world)
        with maximum probability where the evidence holds.
        Most probable explanation (MPE) is MAP where no evidence is present
        i.e., find the world with highest probability where the query is true.
        '''
        self.setup_interface(from_string)
        if len(self.parser.map_id_list) == len(self.interface.prob_facts_dict) and not self.consider_lower_prob and not self.stop_if_inconsistent and not self.normalize_prob:
            print_waring("Brave (upper) MPE can be solved in a faster way using the --solver flag.")
        self.interface.compute_probabilities()
        max_prob, map_state = self.interface.model_handler.get_map_solution(
            self.parser.map_id_list, self.consider_lower_prob)
        if self.interface.normalizing_factor >= 1:
            max_prob = 1
            print_waring("No worlds have > 1 answer sets")

        if self.normalize_prob and self.interface.normalizing_factor != 0:
            max_prob = max_prob / (1 - self.interface.normalizing_factor)

        return max_prob, map_state


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


    @staticmethod
    def print_map_state(prob : float, atoms_list : 'list[list[str]]', n_map_vars : int) -> None:
        '''
        Prints the MAP/MPE state.
        '''
        map_op = len(atoms_list) > 0 and len(atoms_list[0]) == n_map_vars
        map_or_mpe = "MPE" if map_op else "MAP"
        print(f"{map_or_mpe}: {prob}\n{map_or_mpe} states: {len(atoms_list)}")
        for i, el in enumerate(atoms_list):
            print(f"State {i}: {el}")


    @staticmethod
    def print_prob(lp : float, up : float) -> None:
        '''
        Prints the probability values.
        '''
        if lp == up:
            print(f"Lower probability == upper probability for the query: {lp}")
        else:
            print(f"Lower probability for the query: {lp}")
            print(f"Upper probability for the query: {up}")


    @staticmethod
    def remove_dominated_explanations(abd_exp : 'list[list[str]]') -> 'list[set[str]]':
        '''
        Removes the dominated explanations, used in abduction.
        '''
        ls : 'list[set[str]]' = []
        for exp in abd_exp:
            e : 'set[str]' = set()
            for el in exp:
                if not el.startswith('not') and el != 'q':
                    if el.startswith('abd_'):
                        e.add(el[4:])
                    else:
                        e.add(el)
            ls.append(e)

        for i, el in enumerate(ls):
            for j in range(i + 1, len(ls)):
                if len(el) > 0:
                    if el.issubset(ls[j]):
                        ls[j] = set()  # type: ignore

        return ls


    @staticmethod
    def print_result_abduction(lp: float, up: float, abd_exp: 'list[list[str]]', upper : bool = False) -> None:
        '''
        Prints the result for abduction.
        '''
        abd_exp_no_dup = Pasta.remove_dominated_explanations(abd_exp)
        # abd_exp_no_dup = abd_exp
        if len(abd_exp_no_dup) > 0 and up != 0:
            if upper:
                print(f"Upper probability for the query: {up}")
            else:
                Pasta.print_prob(lp, up)

        n_exp = sum(1 for ex in abd_exp_no_dup if len(ex) > 0)
        print(f"Abductive explanations: {n_exp}")

        index = 0
        for el in abd_exp_no_dup:
            if len(el) > 0:
                print(f"Explanation {index}")
                index = index + 1
                print(sorted(el))


def main():
    command_parser = argparse.ArgumentParser(description=pasta_description, epilog=examples_strings)
    command_parser.add_argument("filename", help="Program to analyse", type=str)
    command_parser.add_argument("-q", "--query", help="Query", type=str)
    command_parser.add_argument("-e", "--evidence", help="Evidence", type=str, default="")
    command_parser.add_argument("-v", "--verbose", help="Verbose mode, default: false", action="store_true")
    command_parser.add_argument("--pedantic", help="Pedantic mode (prints the converted program and all the worlds), default: false", action="store_true")
    command_parser.add_argument("--approximate", help="Compute approximate probability", action="store_true")
    command_parser.add_argument("--samples", help="Number of samples, default 1000", type=int, default=1000)
    command_parser.add_argument("--mh", help="Use Metropolis Hastings sampling", action="store_true", default=False)
    command_parser.add_argument("--gibbs", help="Use Gibbs Sampling sampling", action="store_true", default=False)
    command_parser.add_argument("--block", help="Set the block value for Gibbs sampling", type=int, default=1)
    command_parser.add_argument("--rejection", help="Use rejection Sampling sampling", action="store_true", default=False)
    command_parser.add_argument("--pl", help="Parameter learning", action="store_true", default=False)
    command_parser.add_argument("--abduction", help="Abduction", action="store_true", default=False)
    command_parser.add_argument("--map", help="MAP (MPE) inference", action="store_true", default=False)
    command_parser.add_argument("--upper", help="Select upper probability for MAP and abduction", action="store_true", default=False)
    command_parser.add_argument("--minimal", "-nm", help="Compute the minimal set of probabilistic facts", action="store_true", default=False)
    command_parser.add_argument("--normalize", help="Normalize the probability if some worlds have no answer sets", action="store_true", default=False)
    command_parser.add_argument("--stop-if-inconsistent", "-sif", help="Raise an error if some worlds have no answer sets (and lists them)", action="store_true", default=True)
    command_parser.add_argument("--solver", help="Uses an ASP solver for the task", action="store_true", default=False)
    command_parser.add_argument("--one", help="Compute only 1 solution for MAP. Currently has no effects", action="store_true", default=False)
    command_parser.add_argument("--xor", help="Uses XOR constraints for approximate inference", action="store_true", default=False)
    command_parser.add_argument("--alpha", help="Constant for approximate inferece with XOR constraints. Default = 0.004", type=float, default=0.004)
    command_parser.add_argument("--delta", help="Accuracy for approximate inferece with XOR constraints. Default = 2", type=float, default=2)
    command_parser.add_argument("-dt", help="Decision theory", action="store_true", default=False)
    command_parser.add_argument("-k", help="k-credal semantics", type=int, choices=range(1,100), default=100)

    args = command_parser.parse_args()

    if args.rejection or args.mh or args.gibbs:
        args.approximate = True
    if args.dt:
        print_error_and_exit("Not yet implemented")
    if args.k != 100:
        print_error_and_exit("Not yet implemented")
    if args.map and args.solver and not args.upper:
        print_waring("Trying to compute the upper MPE state")
        args.upper = True
    if args.solver:
        args.minimal = False
    if args.minimal and args.stop_if_inconsistent:
        print_waring("The program may be inconsistent")
        args.stop_if_inconsistent = False
    if args.stop_if_inconsistent:
        args.minimal = False

    pasta_solver = Pasta(args.filename, 
                         args.query, 
                         args.evidence, 
                         args.verbose, 
                         args.pedantic,
                         args.samples, 
                         not args.upper, 
                         args.minimal, 
                         args.normalize, 
                         args.stop_if_inconsistent, 
                         args.one, 
                         args.xor, 
                         args.k)

    if args.abduction:
        lower_p, upper_p, abd_explanations = pasta_solver.abduction()
        Pasta.print_result_abduction(lower_p, upper_p, abd_explanations, args.upper)
    elif args.xor:
        lower_p, upper_p = pasta_solver.approximate_solve_xor(args)
        Pasta.print_prob(lower_p, upper_p)
    elif args.approximate:
        lower_p, upper_p = pasta_solver.approximate_solve(args)
        Pasta.print_prob(lower_p, upper_p)
    elif args.pl:
        pasta_solver.parameter_learning()
    elif args.map:
        if args.upper and not (args.normalize or args.stop_if_inconsistent) and args.solver:
            pasta_solver.for_asp_solver = True
            max_p, atoms_list_res = pasta_solver.upper_mpe_inference()
        else:
            max_p, atoms_list_res = pasta_solver.map_inference()
        Pasta.print_map_state(max_p, atoms_list_res, len(pasta_solver.interface.prob_facts_dict))
    elif args.dt:
        lower_p, upper_p, utility_atoms = pasta_solver.decision_theory()
    else:
        lower_p, upper_p = pasta_solver.inference()
        Pasta.print_prob(lower_p, upper_p)


if __name__ == "__main__":
    main()
