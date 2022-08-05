""" Main module for the PASTA solver """
import sys

import argparse

# from pasta.pasta_parser import PastaParser
from pasta_parser import PastaParser
# import pasta_parser
from asp_interface import AspInterface
# import asp_interface
from utils import print_error_and_exit, print_waring

import learning_utilities


examples_string_exact = "python3 pasta_solver.py \
    ../../examples/bird_4.lp \
    --query=\"fly(1)\""
examples_string_exact_evidence = "python3 pasta_solver.py \
    ../../examples/bird_4.lp \
    --query=\"fly(1)\" \
    --evidence=\"bird(1)\""
examples_string_approximate = "python3 pasta_solver.py \
    ../../examples/bird_4.lp \
    --query=\"fly(1)\" \
    --approximate"
examples_string_approximate_rej = "python3 pasta_solver.py \
    ../../examples/bird_4.lp \
    --query=\"fly(1)\" \
    --evidence=\"bird(1)\" --rejection"
examples_strings = "Examples:\n\n" + examples_string_exact + "\n\n" + examples_string_exact_evidence + \
    "\n\n" + examples_string_approximate + "\n\n" + examples_string_approximate_rej

pasta_description = "PASTA: Probabilistic Answer Set programming for STAtistical probabilities"


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
        cautious : bool = True,
        no_minimal : bool = False,
        normalize_prob : bool = False,
        stop_if_inconsistent : bool = False,
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
        self.cautious = cautious
        self.no_minimal = no_minimal
        self.normalize_prob = normalize_prob
        self.stop_if_inconsistent = stop_if_inconsistent
        self.interface : AspInterface
        self.parser : PastaParser


    @staticmethod
    def check_lp_up(lp : float, up : float) -> None:
        '''
        Checks whether lp =< up
        '''
        if (lp > up) or (int(lp * 10e8) > 10e8) or (int(up * 10e8) > 10e8):
            print("Error in computing probabilities")
            print(f"Lower: {lp:.8f}")
            print(f"Upper: {up:.8f}")
            sys.exit()


    def parameter_learning(self, from_string : str = "") -> None:
        '''
        Parameter learning
        '''
        self.parser = PastaParser(self.filename)
        training_set, test_set, program, prob_facts_dict, offset = self.parser.parse_input_learning(from_string)
        interpretations_to_worlds = learning_utilities.learn_parameters(training_set, test_set, program, prob_facts_dict, offset, not self.cautious, self.verbose)
        learning_utilities.test_results(test_set, interpretations_to_worlds, prob_facts_dict, program, offset)


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
            self.samples
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
        self.parser = PastaParser(
            self.filename, self.query, self.evidence)
        self.parser.parse(from_string)

        if self.verbose:
            print("Parsed program")

        if self.no_minimal:
            content_find_minimal_set = []
        else:
            content_find_minimal_set = self.parser.get_content_to_compute_minimal_set_facts()

        asp_program = self.parser.get_asp_program()

        if not self.cautious:
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
            normalize_prob=self.normalize_prob
        )

        exec_time = self.interface.get_minimal_set_facts()

        if self.verbose:
            print(f"Computed cautious consequences in {exec_time} seconds")
            if self.pedantic:
                print("--- Minimal set of probabilistic facts ---")
                print(self.interface.cautious_consequences)
                print("---")

        if self.pedantic:
            print("--- Asp program ---")
            self.interface.print_asp_program()
            print("---")
            print("--- Program to find minimal sets ---")
            for e in content_find_minimal_set:
                print(e)
            print("---")


    def abduction(self, from_string: str = "") -> 'tuple[float,float,list[list[str]]]':
        '''
        Probabilistic and deterministic abduction
        '''
        self.setup_interface(from_string)
        self.interface.abduction()
        lp = self.interface.lower_probability_query
        up = self.interface.upper_probability_query

        self.check_lp_up(lp, up)

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

        self.check_lp_up(lp, up)

        return lp, up


    def map_inference(self, from_string : str = "") -> 'tuple[float,list[list[str]]]':
        '''
        Maximum a posteriori (MAP) inference: find the state (world)
        with maximum probability where the evidence holds.
        Most probable explanation (MPE) is MAP where no evidence is present
        i.e., find the world with highest probability.
        '''
        self.setup_interface(from_string)
        self.interface.compute_probabilities()
        return self.interface.model_handler.get_map_solution(self.parser.map_id_list, self.cautious)


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
    def print_result_abduction(lp: float, up: float, abd_exp: 'list[list[str]]', brave : bool = False) -> None:
        '''
        Prints the result for abduction.
        '''
        abd_exp_no_dup = Pasta.remove_dominated_explanations(abd_exp)
        # abd_exp_no_dup = abd_exp
        if len(abd_exp_no_dup) > 0 and up != 0:
            if brave:
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


if __name__ == "__main__":
    command_parser = argparse.ArgumentParser(
        description=pasta_description, epilog=examples_strings)
    command_parser.add_argument("filename", help="Program to analyse", type=str)
    command_parser.add_argument("-q", "--query", help="Query", type=str)
    command_parser.add_argument("-e", "--evidence", help="Evidence", type=str, default="")
    command_parser.add_argument("-v", "--verbose", help="Verbose mode, default: false", action="store_true")
    command_parser.add_argument("--pedantic", help="Pedantic mode, default: false", action="store_true")
    command_parser.add_argument("--approximate", help="Compute approximate probability", action="store_true")
    command_parser.add_argument("--samples", help="Number of samples, default 1000", type=int, default=1000)
    command_parser.add_argument("--mh", help="Use Metropolis Hastings sampling", action="store_true", default=False)
    command_parser.add_argument("--gibbs", help="Use Gibbs Sampling sampling", action="store_true", default=False)
    command_parser.add_argument("--block", help="Set the block value for Gibbs sampling", type=int, default=1)
    command_parser.add_argument("--rejection", help="Use rejection Sampling sampling", action="store_true", default=False)
    command_parser.add_argument("--pl", help="Parameter learning", action="store_true", default=False)
    command_parser.add_argument("--abduction", help="Abduction", action="store_true", default=False)
    command_parser.add_argument("--map", help="MAP (MPE) inference", action="store_true", default=False)
    command_parser.add_argument("--brave", help="Select upper probability (brave) for MAP and abduction", action="store_true", default=False)
    command_parser.add_argument("--no-minimal", help="Do not compute the minimal set of probabilistic facts", action="store_true", default=False)
    command_parser.add_argument("--normalize", help="Normalize the probability if some worlds do not have answer set", action="store_true", default=False)
    command_parser.add_argument("--stop-if-inconsistent", help="Raise an error if a world without answer sets is found", action="store_true", default=False)

    args = command_parser.parse_args()

    pasta_solver = Pasta(args.filename, args.query, args.evidence, args.verbose, args.pedantic,
                         args.samples, not args.brave, args.no_minimal, args.normalize, args.stop_if_inconsistent)

    if args.abduction is True:
        lower_p, upper_p, abd_explanations = pasta_solver.abduction()
        Pasta.print_result_abduction(lower_p, upper_p, abd_explanations, args.brave)
    elif args.approximate or args.rejection or args.mh or args.gibbs is True:
        lower_p, upper_p = pasta_solver.approximate_solve(args)
        Pasta.print_prob(lower_p, upper_p)
    elif args.pl is True:
        pasta_solver.parameter_learning()
    elif args.map is True:
        max_p, atoms_list_res = pasta_solver.map_inference()
        Pasta.print_map_state(max_p, atoms_list_res, len(pasta_solver.interface.prob_facts_dict))
    else:
        lower_p, upper_p = pasta_solver.inference()
        Pasta.print_prob(lower_p, upper_p)
