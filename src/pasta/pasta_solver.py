import sys

import argparse

# from pasta.pasta_parser import PastaParser
from pasta_parser import PastaParser
# import pasta_parser
from asp_interface import AspInterface
# import asp_interface
from utils import print_error_and_exit


examples_string_exact = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\""
examples_string_exact_evidence = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --evidence=\"bird(1)\""
examples_string_approximate = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --approximate"
examples_string_approximate_rej = "python3 pasta.py ../../examples/bird_4.lp --query=\"fly(1)\" --evidence=\"bird(1)\" --rejection"

examples_strings = "Examples:\n\n" + examples_string_exact + "\n\n" + examples_string_exact_evidence + "\n\n" + examples_string_approximate + "\n\n" + examples_string_approximate_rej

pasta_description = "PASTA: Probabilistic Answer Set programming for STAtistical probabilities"


class Pasta:
    def __init__(
        self,
        filename : str,
        query : str,
        evidence : str = "",
        verbose : bool = False,
        pedantic : bool = False,
        samples : int = 1000
        ) -> None:
        self.filename = filename
        self.query = query
        self.evidence = evidence
        self.verbose = verbose
        self.pedantic = pedantic
        if pedantic is True:
            self.verbose = True
        self.samples = samples
        self.interface : AspInterface

    @staticmethod
    def check_lp_up(lp : float, up : float) -> None:
        '''
        Checks whether lp =< up
        '''
        if (lp > up) or (int(lp * 10e8) > 10e8) or (int(up * 10e8) > 10e8):
            print("Error in computing probabilities")
            print("Lower: " + '{:.8f}'.format(lp))
            print("Upper: " + '{:.8f}'.format(up))
            sys.exit()


    def parameter_learning(self) -> None:
        pass


    def approximate_solve(self, args : argparse.Namespace, from_string : str = "") -> 'tuple[float,float]':
        '''
        Inference through sampling
        '''
        program_parser = PastaParser(self.filename, self.query, self.evidence)
        program_parser.parse_approx(from_string)
        asp_program = program_parser.get_asp_program()

        self.interface = AspInterface(
            program_parser.probabilistic_facts,
            asp_program,
            self.evidence,
            [],
            program_parser.abducibles,
            self.verbose,
            self.pedantic
        )

        if self.evidence == "":
            lp, up = self.interface.sample_query()
        else:
            if args.rejection is True:
                lp, up = self.interface.rejection_sampling()
            elif args.mh is True:
                lp, up = self.interface.mh_sampling()
            elif args.gibbs is True:
                lp, up = self.interface.gibbs_sampling(args.block)
            else:
                lp = 0
                up = 0
                print_error_and_exit("Sampling method found")
                
        return lp, up


    def setup_interface(self, from_string : str = "") -> None:
        '''
        Setup clingo interface
        '''
        program_parser = PastaParser(
            self.filename, self.query, self.evidence)
        program_parser.parse(from_string)

        if self.verbose:
            print("Parsed program")

        content_find_minimal_set = program_parser.get_content_to_compute_minimal_set_facts()
        asp_program = program_parser.get_asp_program()

        self.interface = AspInterface(
            program_parser.probabilistic_facts,
            asp_program,
            self.evidence,
            content_find_minimal_set,
            program_parser.abducibles,
            self.verbose,
            self.pedantic
        )

        exec_time = self.interface.get_minimal_set_facts()

        if self.verbose:
            print("Computed cautious consequences in %s seconds" % (exec_time))
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

        self.check_lp_up(lp, up)

        return lp, up


    def map_inference(self, from_string: str = "")  -> 'tuple[float,list[str]]':
        '''
        Maximum a posteriori (MAP) inference: find the state (world) 
        with maximum probability where the evidence holds.
        Most probable explanation (MPE) is MAP where no evidence is present
        i.e., find the world with highest probability.
        '''
        self.setup_interface(from_string)
        # TODO: this is a quick test, all the facts
        # are considered map. Add a list of index that
        # indicates the indexes of the map facts in the list
        # of all the probabilistic facts. Then, retrieve
        # the state when only these are considered. In this way,
        # i do not need another complex class
        self.interface.compute_probabilities()

        max_prob : float = 0.0
        w_id : str = ""
        for el in self.interface.model_handler.worlds_dict:
            w = self.interface.model_handler.worlds_dict[el]
            if w.prob > max_prob and w.model_query_count > 0 and w.model_not_query_count == 0:
                max_prob = w.prob
                w_id = el
        
        atoms_list = self.interface.model_handler.get_map_word_from_id(w_id)

        return max_prob, atoms_list


    @staticmethod
    def print_map_state(prob : float, atoms_list : 'list[str]') -> None:
        '''
        Prints the MAP state
        '''
        print(f"MAP: {prob}")
        print(atoms_list)


    @staticmethod
    def print_prob(lp : float, up : float) -> None:
        '''
        Print the probability values
        '''
        if lp == up:
            print(f"Lower probability == upper probability for the query: {lp}")
        else:
            print(f"Lower probability for the query: {lp}")
            print(f"Upper probability for the query: {up}")


    @staticmethod
    def remove_dominated_explanations(abd_exp : 'list[list[str]]') -> 'list[set[str]]':
        ls : list[set[str]] = []
        for exp in abd_exp:
            e : set[str] = set()
            for el in exp:
                if not el.startswith('not') and el != 'q':
                    if el.startswith('abd_'):
                        e.add(el[4:])
                    else:
                        e.add(el)
            ls.append(e)

        for i in range(0, len(ls)):
            for j in range(i + 1, len(ls)):
                if len(ls[i]) > 0:
                    if ls[i].issubset(ls[j]):
                        ls[j] = set()  # type: ignore

        return ls


    @staticmethod
    def print_result_abduction(lp: float, up: float, abd_exp: 'list[list[str]]') -> None:
        abd_exp_no_dup = Pasta.remove_dominated_explanations(abd_exp)
        # abd_exp_no_dup = abd_exp
        if len(abd_exp_no_dup) > 0 and up != 0:
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
    command_parser.add_argument("-pl", help="Parameter learning", action="store_true", default=False)
    command_parser.add_argument("--abduction", help="Abduction", action="store_true", default=False)
    command_parser.add_argument("--map", help="MAP (MPE) inference", action="store_true", default=False)

    args = command_parser.parse_args()

    pasta_solver = Pasta(args.filename, args.query, args.evidence, args.verbose, args.pedantic, args.samples)

    args.approximate = False

    if args.abduction is True:
        lp, up, abd_explanations = pasta_solver.abduction()
        Pasta.print_result_abduction(lp, up, abd_explanations)
    elif args.approximate is True:
        lp, up = pasta_solver.approximate_solve(args)
        Pasta.print_prob(lp, up)
    elif args.pl is True:
        pasta_solver.parameter_learning()
    elif args.map is True:
        max_p, atoms_list = pasta_solver.map_inference()
        Pasta.print_map_state(max_p, atoms_list)
    else:
        lp, up = pasta_solver.inference()
        Pasta.print_prob(lp, up)
