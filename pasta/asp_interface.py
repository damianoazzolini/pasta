""" Module implementing the connection to clingo """

import clingo
import math
import random
import time

from . import utils
from .models_handler import ModelsHandler
from .optimizable import compute_optimal_probability
from .reducible import reduce_pasp_up

def reconstruct_atom(atm) -> str:  # type: ignore
    '''
    Reconstructs a probabilistic fact from a clingo representation
    of its. This is needed since a(1) is stored as a with 1 argument
    and not as a(1) (string)
    '''
    s = f"{atm.symbol.name}("  # type: ignore
    for arg in atm.symbol.arguments:  # type: ignore
        s = s + str(arg) + ","  # type: ignore
    if s[len(s) - 1] == '(':
        return s[:-1]
    return s[:-1] + ')'


def pick_random_index(block : int, w_id : str) -> 'list[int]':
    '''
    Pick a random index, used in Gibbs sampling.
    '''
    return sorted(set([random.randint(0, len(w_id) - 1) for _ in range(0, block)]))


def compute_conditional_lp_up(
    n_lower_qe: int,
    n_upper_qe: int,
    n_lower_nqe: int,
    n_upper_nqe: int,
    n_samples: int
    ) -> 'tuple[float,float]':
    '''
    Computes the lower and upper conditional probabilities using the
    formulas:
    lower P(q | e) = lower P(q,e) / (lower P(q,e) + upper P(not q,e))
    upper P(q | e) = upper P(q,e) / (upper P(q,e) + lower P(not q,e))
    '''

    lower_q_e = n_lower_qe / n_samples
    upper_q_e = n_upper_qe / n_samples
    lower_not_q_e = n_lower_nqe / n_samples
    upper_not_q_e = n_upper_nqe / n_samples

    lp = lower_q_e / \
        (lower_q_e + upper_not_q_e) if (lower_q_e + upper_not_q_e) > 0 else 0
    up = upper_q_e / \
        (upper_q_e + lower_not_q_e) if (upper_q_e + lower_not_q_e) > 0 else 0

    return lp, up


class AspInterface:
    '''
    Parameters:
        - content: list with the program
    '''

    def __init__(self,
        probabilistic_facts : 'dict[str,float]',
        asp_program : 'list[str]',
        evidence : str = "",
        program_minimal_set : 'list[str]' = [],
        abducibles_list : 'list[str]' = [],
        verbose : bool = False,
        pedantic : bool = False,
        n_samples : int = 1000,
        stop_if_inconsistent : bool = False,
        normalize_prob : bool = False,
        xor : bool = False,
        decision_atoms_list : 'list[str]' = [],
        utilities_dict : 'dict[str,float]' = {},
        upper : bool = False,
        n_probabilistic_ics : int = 0,
        k_credal: int = 100,
        # constraints : 'list[str]' = [], # for the optimizable task
        # objective_function : str = "", # for the optimizable task
        optimizable_facts : 'dict[str,tuple[float,float]]' = {}, # for the optimizable task
        reducible_facts : 'dict[str,float]' = {} # for the reducible task
        ) -> None:
        self.cautious_consequences : 'list[str]' = []
        self.program_minimal_set : 'list[str]' = sorted(set(program_minimal_set))
        self.asp_program : 'list[str]' = sorted(set(asp_program))
        self.lower_probability_query : float = 0
        self.upper_probability_query : float = 0
        self.upper_probability_evidence : float = 0
        self.lower_probability_evidence : float = 0
        self.evidence : str = evidence
        self.abducibles_list : 'list[str]' = abducibles_list
        self.computed_models : int = 0
        self.abductive_explanations : 'list[list[str]]' = []
        self.verbose : bool = verbose
        self.pedantic : bool = pedantic
        self.n_samples : int = n_samples
        self.inconsistent_worlds : 'dict[str,float]' = {}
        self.prob_facts_dict : 'dict[str,float]' = probabilistic_facts
        self.stop_if_inconsistent : bool = stop_if_inconsistent
        self.normalize_prob : bool = normalize_prob
        self.normalizing_factor : float = 1
        self.xor: bool = xor
        self.decision_atoms_selected : 'list[str]' = []
        self.utility : 'list[float]' = [-math.inf,-math.inf]
        self.decision_atoms_list: 'list[str]' = decision_atoms_list
        self.utilities_dict : 'dict[str,float]' = utilities_dict
        self.upper : bool = upper
        self.n_probabilistic_ics : int = n_probabilistic_ics
        self.k_credal : int = k_credal

        # for the optimizable task
        # self.constraints : 'list[str]' = constraints
        # self.objective_function : str = objective_function
        self.optimizable_facts : 'dict[str,tuple[float,float]]' = optimizable_facts
        self.reducible_facts : 'dict[str,float]' = reducible_facts

        self.model_handler : ModelsHandler = \
            ModelsHandler(
                self.prob_facts_dict,
                self.evidence,
                self.abducibles_list,
                self.decision_atoms_list,
                self.utilities_dict
            )


    def compute_minimal_set_facts(self) -> None:
        '''
        Compute the minimal set of probabilistic/abducible facts
        needed to make the query true. This operation is performed
        only if there is not evidence.
        Cautious consequences: clingo <filename> -e cautious
        '''
        ctl = self.init_clingo_ctl(["--enum-mode=cautious", "-Wnone"], self.program_minimal_set)

        temp_cautious = []
        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                # i need only the last one
                temp_cautious = str(m).split(' ')  # type: ignore
            handle.get()  # type: ignore

        for el in temp_cautious:
            # if el != '' and (el.split(',')[-2] + ')' if el.count(',') > 0 else el.split('(')[0]) in self.probabilistic_facts:
            if el != '':
                self.cautious_consequences.append(el)


    def compute_probabilities(self) -> None:
        '''
        Computes the lower and upper bound for the query
        '''
        clingo_arguments : 'list[str]' = ["0","-Wnone"]
        clingo_arguments.append("--project")
        clauses = self.asp_program
        for c in self.cautious_consequences:
            clauses.append(f':- not {c}.')

        ctl = self.init_clingo_ctl(clingo_arguments, clauses)

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                self.model_handler.add_value(str(m))  # type: ignore
                self.computed_models = self.computed_models + 1
            handle.get()   # type: ignore

        self.normalizing_factor = 1
        # print(self.model_handler.worlds_dict)

        if len(self.model_handler.worlds_dict) != 2**len(self.prob_facts_dict):
            if len(self.model_handler.worlds_dict) == 0 and len(self.prob_facts_dict) > 0:
                self.lower_probability_query = 0
                self.upper_probability_query = 0
                utils.print_pathological_program()
                return

            missing = []
            if self.pedantic:
                ks = sorted(self.model_handler.worlds_dict.keys())
                l : 'list[int]' = []
                for el in ks:
                    if el != '':
                        l.append(int(el,2))
                missing = sorted(set(range(0, 2**len(self.prob_facts_dict))).difference(l), key=lambda x: bin(x)[2:].count('1'))

                ntw = len(self.model_handler.worlds_dict) + 2**(len(self.prob_facts_dict) - len(self.cautious_consequences))
                nw = 2**len(self.prob_facts_dict)

                # TODO: check this case
                if len(self.cautious_consequences) > 0 and (ntw != nw) and not self.xor and not self.upper:
                    utils.print_inconsistent_program(self.stop_if_inconsistent)
            
            if self.stop_if_inconsistent and not self.normalize_prob and len(self.prob_facts_dict) > 0:
                res = ""
                for el in missing:
                    s = "0"*(len(self.prob_facts_dict) - len(bin(el)[2:])) + bin(el)[2:]
                    i = 0
                    res = res + s + "{ "
                    for el in self.prob_facts_dict:
                        if s[i] == '1':
                            res += el + " "
                        i = i + 1
                    res += "}\n"
                if self.pedantic:
                    utils.print_error_and_exit(f"Found {len(missing)} worlds without answer sets: {missing}\n{res[:-1]}.")
                else:
                    utils.print_error_and_exit(f"Found {2**len(self.prob_facts_dict) - len(self.model_handler.worlds_dict)} worlds without answer sets.")

            norm_fact = sum([x.prob for x in self.model_handler.worlds_dict.values()])
            
            if self.normalize_prob:
                self.normalizing_factor = norm_fact

            if self.pedantic:
                # print(f"n missing {len(missing)}")
                # print(self.inconsistent_worlds)
                if self.normalize_prob:
                    print(f"Normalizing factor: {self.normalizing_factor}")
                elif not self.stop_if_inconsistent:
                    print(f"P(inc) = {1 - norm_fact}")

        if self.pedantic:
            print(utils.RED + "lp" + utils.END + utils.YELLOW + " up" + utils.END)
            for el in self.prob_facts_dict:
                print(f"{el} : {self.prob_facts_dict[el]}")
            print("_"*50)
            for el in self.prob_facts_dict:
                print(el, end="\t")
            print("#pf", end="\t")
            print("LP/UP\tProbability")
            lp_count = 0
            up_count = 0
            for el in sorted(self.model_handler.worlds_dict, key= lambda x: x.count('1')):
                for val in el:
                    print(f"{val}", end="\t")
                print(f"{el.count('1')}", end="\t")
                if self.model_handler.worlds_dict[el].model_query_count > 0 and \
                    self.model_handler.worlds_dict[el].model_not_query_count == 0:
                    print(utils.RED + "LP\t", end = "")
                    lp_count = lp_count + 1
                elif self.model_handler.worlds_dict[el].model_query_count > 0 and \
                    self.model_handler.worlds_dict[el].model_not_query_count > 0:
                    print(utils.YELLOW + "UP\t", end="")
                    up_count = up_count + 1
                else:
                    print("-\t", end="")
                print(self.model_handler.worlds_dict[el].prob, end="")
                if self.model_handler.worlds_dict[el].model_query_count > 0:
                    print(utils.END)
                else:
                    print("")
            print(f"Total number of worlds that contribute to the probability: {lp_count + up_count}")
            print(f"Only LP: {lp_count}, Only UP: {up_count}")
        
        self.lower_probability_query, self.upper_probability_query = self.model_handler.compute_lower_upper_probability(self.k_credal)
        if self.normalizing_factor == 0:
            self.lower_probability_query = 1
            self.upper_probability_query = 1
            # utils.print_warning("No worlds have > 1 answer sets")
        else:
            self.lower_probability_query = self.lower_probability_query / self.normalizing_factor
            self.upper_probability_query = self.upper_probability_query / self.normalizing_factor


    def compute_mpe_asp_solver(self, one : bool = False) -> 'tuple[str,bool]':
        '''
        Computes the upper MPE state by using an ASP solver.
        Assumes that every world has at least one answer set.
        '''
        ctl = self.init_clingo_ctl(["-Wnone","--opt-mode=opt","--models=0", "--output-debug=none"])
        opt : str = " "
        unsat : bool = True
        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                unsat = False
                opt = str(m)  # type: ignore
            handle.get()   # type: ignore

        return opt, unsat


    def sample_world(self, randomly : bool = False) -> 'tuple[dict[str,bool],str]':
        '''
        Samples a world for approximate probability computation.
        '''
        w_id: 'dict[str,bool]' = {}
        w_id_key: str = ""

        for key in self.prob_facts_dict:
            comp = 0.5 if randomly else self.prob_facts_dict[key]
            if random.random() < comp:
                w_id[key] = True
                w_id_key = w_id_key + "T"
            else:
                w_id[key] = False
                w_id_key = w_id_key + "F"

        return w_id, w_id_key


    def resample(self, i : int) -> 'tuple[str,str]':
        '''
        Resamples a facts. Used in Gibbs sampling.
        '''
        key : str = ""
        for k in self.prob_facts_dict:
            key = k
            i = i - 1
            if i < 0:
                break

        if random.random() < self.prob_facts_dict[key]:
            return 'T', key
        return 'F', key


    @staticmethod
    def assign_T_F_and_get_count(ctl : clingo.Control, w_assignments: 'dict[str,bool]') -> 'tuple[int,int,int,int]':
        '''
        It does what it is specified in its name.
        '''
        i = 0
        for atm in ctl.symbolic_atoms:
            if atm.is_external:
                ctl.assign_external(atm.literal, w_assignments[reconstruct_atom(atm)])
                i = i + 1

        qe_count = 0
        qe_false_count = 0
        nqe_count = 0
        nqe_false_count = 0

        # I can have: qe or qe_false, nqe or nqe_false
        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                m1 = str(m).split(' ')  # type: ignore
                if 'qe' in m1:
                    qe_count = qe_count + 1
                else:
                    qe_false_count = qe_false_count + 1
                if 'nqe' in m1:
                    nqe_count = nqe_count + 1
                else:
                    nqe_false_count = nqe_false_count + 1

        return qe_count, qe_false_count, nqe_count, nqe_false_count


    @staticmethod
    def assign_T_F_and_check_if_evidence(ctl : clingo.Control, w_assignments: 'dict[str,bool]') -> bool:
        '''
        Assigns T or F to facts and checks whether q and e or not q and e
        is true.
        Used in Gibbs sampling.
        '''

        for atm in ctl.symbolic_atoms:
            if atm.is_external:
                ctl.assign_external(atm.literal, w_assignments[reconstruct_atom(atm)])

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                m1 = str(m).split(' ')  # type: ignore
                if 'qe' in m1 or 'nqe' in m1:
                    return True

        return False


    @staticmethod
    def get_val_or_compute_and_update_dict(
        sampled : 'dict[str,list[int]]',
        ctl : clingo.Control,
        w_assignments: 'dict[str,bool]',
        w_id : str
        ) -> 'tuple[int,int,int,int]':
        '''
        If the world is has been already considered, retrieve it; otherwise
        compute its contribution.
        Used for sampling
        '''
        if w_id in sampled:
            return sampled[w_id][0], sampled[w_id][1], sampled[w_id][2], sampled[w_id][3]

        qe_count, qe_false_count, nqe_count, nqe_false_count = AspInterface.assign_T_F_and_get_count(ctl, w_assignments)

        lower_qe = (1 if qe_false_count == 0 else 0)
        upper_qe = (1 if qe_count > 0 else 0)
        lower_nqe = (1 if nqe_false_count == 0 else 0)
        upper_nqe = (1 if nqe_count > 0 else 0)

        # update sampled table
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe]
        sampled[w_id] = [
            lower_qe,
            upper_qe,
            lower_nqe,
            upper_nqe
        ]

        return lower_qe, upper_qe, lower_nqe, upper_nqe


    def init_clingo_ctl(self, clingo_arguments : 'list[str]', clauses : 'list[str]' = []) -> 'clingo.Control':
        '''
        Init clingo and grounds the program
        '''
        ctl = clingo.Control(clingo_arguments)
        lines = self.asp_program if len(clauses) == 0 else clauses
        try:
            for clause in lines:
                ctl.add('base', [], clause)
            ctl.ground([("base", [])])
        except RuntimeError:
            utils.print_error_and_exit('Syntax error, parsing failed.')

        return ctl

    def mh_sampling(self, dummy: bool = True) -> 'tuple[float, float]':
        '''
        MH sampling.
        The argument dummy is for multiprocessing, to use imap
        '''
        # each element has the structure
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, T_count]
        sampled : 'dict[str,list[int]]' = {}

        ctl = self.init_clingo_ctl(["0", "--project"])

        n_samples = self.n_samples

        w_assignments, w_id = self.sample_world()
        t_count = w_id.count('T')
        previous_t_count = t_count if t_count > 0 else 1

        n_lower_qe : int = 0
        n_upper_qe : int = 0
        n_lower_nqe : int = 0
        n_upper_nqe : int = 0

        k : int = 0

        current_t_count : int = 1

        previous_t_count : int = 1

        while k < n_samples:
            w_assignments, w_id = self.sample_world()
            k = k + 1

            if w_id in sampled:
                current_t_count = sampled[w_id][4]

                if random.random() < min(1, current_t_count / previous_t_count):
                    n_lower_qe = n_lower_qe + sampled[w_id][0]
                    n_upper_qe = n_upper_qe + sampled[w_id][1]
                    n_lower_nqe = n_lower_nqe + sampled[w_id][2]
                    n_upper_nqe = n_upper_nqe + sampled[w_id][3]

                previous_t_count = current_t_count
            else:
                qe_count, qe_false_count, nqe_count, nqe_false_count = AspInterface.assign_T_F_and_get_count(ctl, w_assignments)

                if qe_count > 0 or nqe_count > 0:
                    t_count = w_id.count('T')
                    current_t_count = t_count if t_count > 0 else 1

                    if random.random() < min(1, current_t_count / previous_t_count):
                        # k = k + 1
                        n_lower_qe = n_lower_qe + (1 if qe_false_count == 0 else 0)
                        n_upper_qe = n_upper_qe + (1 if qe_count > 0 else 0)
                        n_lower_nqe = n_lower_nqe + (1 if nqe_false_count == 0 else 0)
                        n_upper_nqe = n_upper_nqe + (1 if nqe_count > 0 else 0)

                        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe]
                        sampled[w_id] = [
                            1 if qe_false_count == 0 else 0,
                            1 if qe_count > 0 else 0,
                            1 if nqe_false_count == 0 else 0,
                            1 if nqe_count > 0 else 0,
                            current_t_count
                        ]

                previous_t_count = current_t_count

        return compute_conditional_lp_up(n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, n_samples)


    def gibbs_sampling(self, block: int) -> 'tuple[float, float]':
        '''
        Gibbs sampling
        '''
        # list of samples for the evidence
        # correspondence str -> bool
        sampled_evidence : 'dict[str,bool]' = {}
        # list of samples for the query
        # each element has the structure
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe]
        sampled_query: 'dict[str,list[int]]' = {}

        ctl = self.init_clingo_ctl(["0", "--project"])

        n_samples = self.n_samples

        n_lower_qe : int = 0
        n_upper_qe : int = 0
        n_lower_nqe : int = 0
        n_upper_nqe : int = 0

        k : int = 0

        ev : bool = False

        w_id : str = ""
        idNew : str = ""
        w_assignments: 'dict[str,bool]' = {}

        while k < n_samples:
            k = k + 1

            # Step 0: sample evidence
            ev = False
            while ev is False:
                w_assignments, w_id = self.sample_world()
                if w_id in sampled_evidence:
                    ev = sampled_evidence[w_id]
                else:
                    ev = AspInterface.assign_T_F_and_check_if_evidence(ctl, w_assignments)
                    sampled_evidence[w_id] = ev

            # Step 1: switch samples but keep the evidence true
            ev = False

            while ev is False:
                # blocked gibbs
                to_resample = pick_random_index(block, w_id)
                idNew = w_id
                for i in to_resample:
                    value, key = self.resample(i)
                    idNew = idNew[:i] + value + idNew[i + 1:]
                    w_assignments[key] = False if value == 'F' else True

                if idNew in sampled_evidence:
                    ev = sampled_evidence[idNew]
                else:
                    ev = AspInterface.assign_T_F_and_check_if_evidence(ctl, w_assignments)
                    sampled_evidence[idNew] = ev

            # step 2: ask query
            lower_qe, upper_qe, lower_nqe, upper_nqe = AspInterface.get_val_or_compute_and_update_dict(sampled_query, ctl, w_assignments, idNew)

            n_lower_qe = n_lower_qe + lower_qe
            n_upper_qe = n_upper_qe + upper_qe
            n_lower_nqe = n_lower_nqe + lower_nqe
            n_upper_nqe = n_upper_nqe + upper_nqe

        return compute_conditional_lp_up(n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, n_samples)


    def rejection_sampling(self, dummy : bool = True) -> 'tuple[float, float]':
        '''
        Rejection Sampling.
        The argument dummy is for multiprocessing, for imap.
        '''
        # each element has the structure
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe]
        sampled = {}

        ctl = self.init_clingo_ctl(["0", "--project"])

        n_lower_qe : int = 0
        n_upper_qe : int = 0
        n_lower_nqe : int = 0
        n_upper_nqe : int = 0

        k : int = 0

        while k < self.n_samples:
            w_assignments, w_id = self.sample_world()
            k = k + 1
            lower_qe, upper_qe, lower_nqe, upper_nqe = AspInterface.get_val_or_compute_and_update_dict(sampled, ctl, w_assignments, w_id)

            n_lower_qe = n_lower_qe + lower_qe
            n_upper_qe = n_upper_qe + upper_qe
            n_lower_nqe = n_lower_nqe + lower_nqe
            n_upper_nqe = n_upper_nqe + upper_nqe

        return compute_conditional_lp_up(n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, self.n_samples)


    def sample_query(self, dummy : bool = True) -> 'tuple[float, float]':
        '''
        Samples the query self.n_samples times.
        dummy is a dummy argument for imap_unordered, that requires
        an iterator as argument.
        '''
        # sampled worlds
        # each element is a list [lower, upper]
        sampled : 'dict[str,list[int]]' = {}

        ctl = self.init_clingo_ctl(["0", "--project"])

        n_lower : int = 0
        n_upper : int = 0
        
        n_inconsistent : int = 0
        
        inc_sampled : 'dict[str,int]' = {}
        if self.pedantic:
            print(f"Taking {self.n_samples} samples")

        # for _ in utils.progressbar(range(self.n_samples), "Computing: ", 40):
        for _ in range(self.n_samples):
            w_assignments, w_id = self.sample_world()

            if w_id in sampled:
                if sampled[w_id][0] == -1 and sampled[w_id][1] == -1:
                    n_inconsistent += 1
                    inc_sampled[w_id] += 1
                else:
                    n_lower = n_lower + sampled[w_id][0]
                    n_upper = n_upper + sampled[w_id][1]
            else:
                for atm in ctl.symbolic_atoms:
                    if atm.is_external:
                        atom = reconstruct_atom(atm)
                        if atom in self.prob_facts_dict:
                            ctl.assign_external(atm.literal, w_assignments[atom])

                upper_count = 0
                lower_count = 0
                with ctl.solve(yield_=True) as handle:  # type: ignore
                    for m in handle:  # type: ignore
                        if str(m) == "q":  # type: ignore
                            upper_count = upper_count + 1
                        else:
                            lower_count = lower_count + 1

                        handle.get()  # type: ignore

                # if lower_count == 0 and upper_count == 0:
                if str(ctl.solve()) == "UNSAT":
                    if not self.normalize_prob:
                        utils.print_inconsistent_program_approx(self.stop_if_inconsistent, w_id)
                    n_inconsistent += 1
                    inc_sampled[w_id] = 1

                if lower_count + upper_count > 0:
                    up = 1 if upper_count > 0 else 0
                    lp = 1 if up and lower_count == 0 else 0
                    n_lower = n_lower + lp
                    n_upper = n_upper + up
                else:
                    lp = -1
                    up = -1

                sampled[w_id] = [lp, up]

        if self.normalize_prob:
            p_inc = n_inconsistent / self.n_samples
            lp_u = n_lower / self.n_samples
            up_u = n_upper / self.n_samples
            return lp_u / (1 - p_inc), up_u / (1 - p_inc)

        return n_lower / self.n_samples, n_upper / self.n_samples


    def check_inconsistency_by_sampling(self, just_test: bool = False) -> 'tuple[list[str],list[str],int]':
        '''
        If just_test = True, then stops as soon as it founds and inconsistent world.
        '''
        inconsistent : 'list[str]' = []
        tested : 'list[str]' = []
        unsat_count : int = 0
        iterations : int = 0

        ctl = self.init_clingo_ctl(["-Wnone"])
        # for _ in range(self.n_samples):
        while True:
            w_assignments, w_id = self.sample_world(True)
            
            if w_id not in tested:
                tested.append(w_id)

            for atm in ctl.symbolic_atoms:
                if atm.is_external:
                    atom = reconstruct_atom(atm)
                    if atom in self.prob_facts_dict:
                        ctl.assign_external(atm.literal, w_assignments[atom])
            if str(ctl.solve()) == "UNSAT":
                unsat_count += 1
                if w_id not in inconsistent:
                    inconsistent.append(w_id)
                if just_test:
                    return tested, inconsistent, iterations

            if len(tested) == 2**len(self.prob_facts_dict):
                return tested, inconsistent, iterations

            iterations += 1


    def extract_best_utility(
            self,
            computed_utilities_list: 'dict[str,tuple[float,float,float]]',
            lower : bool = False
        ) -> 'tuple[tuple[float,float,float],list[str]]':
        '''
        Loops over the utility list and find the best assignment.
        '''
        best : 'list[float]' = [-math.inf,-math.inf,0] # lower utility, upper utility, and unsat worlds
        best_comb : 'list[str]' = []
        for el in computed_utilities_list:
            index = 0 if lower else 1
            if computed_utilities_list[el][index] > best[index] or ((computed_utilities_list[el][index] == best[index]) and (computed_utilities_list[el][1 if lower else 0] == best[1 if lower else 0])):
                best[index] = computed_utilities_list[el][index]
                best[1 if lower else 0] = computed_utilities_list[el][1 if lower else 0]
                best[2] = computed_utilities_list[el][2]
                best_comb = []
                for c, decision in zip(el,self.decision_atoms_list):
                    if int(c) == 1:
                        best_comb.append(decision)
                    else:
                        best_comb.append(f"not {decision}")

        return best, best_comb


    def _evaluate_strategy_dtopt(
            self,
            bin_value : str
        ) -> 'tuple[float,float,float]':
        '''
        Evaluates a strategy: it computes all the words and extract the
        min and max utility.
        '''
        # bin_value = bin(strategy)[2:].zfill(len(self.decision_atoms_list))
        st = ""
        for index, v in enumerate(bin_value):
            if int(v) == 1:
                st += f"{self.decision_atoms_list[int(index)]} "
        if self.verbose:
            print(f"----- Strategy: [{st}] -----")
        # s = "0"*(len(self.decision_atoms_list) - len(bin(el)[2:]))
        # add the constraints for the truth values of the facts
        # constraints : list[str] = []
        for index, v in enumerate(bin_value):
            mode = "" if int(v) == 0 else "not"
            c = f":- {mode} {self.decision_atoms_list[index]}."
            # constraints.append(c)
            self.asp_program.append(c)
        # enumerate the worlds
        all_worlds : 'list[int]' = list(range(0, 2**len(self.prob_facts_dict)))
        w_list : 'list[str]' = list(self.prob_facts_dict.keys())
        lr = 0
        ur = 0
        p_unsat = 0 # probability of the UNSAT worlds
        for w in all_worlds:
            original_prg_constr = self.asp_program.copy()
            
            # compute the encoding of the world
            bin_value_w = bin(w)[2:].zfill(len(self.prob_facts_dict))
            world_probability : float = 1

            # impose the constraint on the probabilistic facts true in the world
            w_str = ""
            for index, v in enumerate(bin_value_w):
                mode = "" if int(v) == 0 else "not"
                if int(v) == 1:
                    w_str += f"{w_list[index]} "
                c = f":- {mode} {w_list[index]}."
                self.asp_program.append(c)
                if int(v) == 1:
                    world_probability *= self.prob_facts_dict[w_list[index]]
                else:
                    world_probability *= (1-self.prob_facts_dict[w_list[index]])
            
            # remove the show statements and insert only the ones on the utility
            self.asp_program = [s for s in self.asp_program if not s.startswith('#show')]
            for el in self.utilities_dict:
                if el.count('(') > 0:
                    self.asp_program.append(f"#show {el.split('(')[0]}/{el.count('(') + el.count(',') }.")
                else:
                    self.asp_program.append(f"#show {el}/0.")


            # enumerate the answer sets
            ctl = self.init_clingo_ctl(["0"])
            # print(f"PROGRAM for strategy [{st}] world [{w_str}]")
            # self.print_asp_program()
            # import sys
            # sys.exit()
            as_list : 'list[list[str]]' = []
            with ctl.solve(yield_=True) as handle:  # type: ignore
                for m in handle:  # type: ignore
                    as_list.append(str(m).split(' '))  # type: ignore
                handle.get()  # type: ignore
                
            if self.pedantic:
                print(f"\tworld [{w_str}] probability {world_probability}")
            # print(as_list)
            # compute the lower and upper bounds
            if len(as_list) >= 1 and as_list[0] != ['']:
                rewards_list_inner : list[float] = []
                for answer_set in as_list:
                    # r : float = 1 # for product
                    r : float = 0
                    for atom in answer_set:
                        if len(atom) > 0:
                            # r *= self.utilities_dict[atom]
                            r += self.utilities_dict[atom]
                    rewards_list_inner.append(r)

                lr_obt = min(rewards_list_inner) * world_probability
                ur_obt = max(rewards_list_inner) * world_probability
                
                if self.pedantic:
                    print(f"Rewards obtained for world [{w_str}]")
                    for as_el,as_rew in zip(as_list,rewards_list_inner):
                        print(f"\t{as_el} -> {as_rew}")
                    print(f"Optimal reward: ({lr_obt},{ur_obt})")

                lr += lr_obt
                ur += ur_obt
                # lb, ub = compute_utility_bounds(as_list, world_probability, self.utilities_dict)
            elif len(as_list) == 0:
                # no answer sets
                p_unsat += world_probability
            if self.pedantic:
                if len(as_list) == 0:
                    print(f"Unsat {w_str}")
                else:
                    print(f"SAT {w_str} but 0")
                # print(f"Rewards obtained for world [{w_str}]: 0")
            self.asp_program = original_prg_constr.copy()
        
        if self.verbose:
            print(f"Strategy: [{st}], Utility: [{lr, ur, p_unsat}]")

        return lr, ur, p_unsat


    def decision_theory_opt(
            self,
            approximate : bool = False,
            samples : int = 1000,
            to_maximize : str = "upper"
        ) -> 'tuple[tuple[float,float,float],list[str]]':
        '''
        Solves decision theory with optimization approaches.
        '''
        def flip_bit(s : str, one : bool = True) -> str:
            '''
            Flips bits in a binary string, used for approximate.
            One flips one random bit, otherwise flips each one
            with probaiblity 0.5
            '''
            st = list(s)
            if one:
                idx = random.randint(0,len(st) - 1)
                st[idx] = '1' if st[idx] == '0' else '0'
            else:
                for idx in range(0, len(st)):
                    if random.random() > 0.5:
                        st[idx] = '1' if st[idx] == '0' else '0'
                    
            return ''.join(st)
        
        ######### BODY
        computed_utilities_dict : 'dict[str,tuple[float,float,float]]' = {}

        if len(self.decision_atoms_list) == 0:
            utils.print_error_and_exit("Specify at least one decision atom.")

        original_prg = self.asp_program.copy()
        
        if not approximate:
            decision_facts_combinations = list(range(0, 2**len(self.decision_atoms_list)))
            # exact: enumerates all the strategies
            for strategy in decision_facts_combinations:
                bin_value = bin(strategy)[2:].zfill(len(self.decision_atoms_list))
                lr, ur, p_unsat = self._evaluate_strategy_dtopt(bin_value)
                computed_utilities_dict[bin_value] = (lr,ur,p_unsat)
                # restore the program, since it is modified in evaluate strategy
                self.asp_program = original_prg.copy()
            
            return self.extract_best_utility(computed_utilities_dict, self.upper)
        else:
            current_strategy = random.randint(0, 2**len(self.decision_atoms_list) - 1)
            bin_value_current_strategy = bin(current_strategy)[2:].zfill(len(self.decision_atoms_list))
            lr, ur, p_unsat = self._evaluate_strategy_dtopt(bin_value_current_strategy)
            # print(f"add: {bin_value_current_strategy} -> {(lr, ur, p_unsat)}")
            computed_utilities_dict[bin_value_current_strategy] = (lr, ur, p_unsat)
            # cleanup - crucial
            self.asp_program = original_prg.copy()
            
            if to_maximize == "upper":
                best_value = ur
            else:
                best_value = lr

            # hill climbing: iteratively improves a strategy
            for i in range(1, samples):
                # selects a strategy
                bin_selected_strategy = flip_bit(bin_value_current_strategy, one=False)
                # print(bin_selected_strategy)
                if not(bin_selected_strategy in computed_utilities_dict):
                    lr, ur, p_unsat = self._evaluate_strategy_dtopt(bin_selected_strategy)

                    if to_maximize == "upper":
                        current_value = ur
                    else:
                        current_value = lr
                    
                    if current_value > best_value:
                        # found a better one
                        best_value = current_value
                        bin_value_current_strategy = bin_selected_strategy
                        if self.verbose:
                            print(f"Improved at iteration {i}")
                    # print(f"add: {bin_selected_strategy} -> {(lr, ur, p_unsat)}")
                    computed_utilities_dict[bin_selected_strategy] = (lr, ur, p_unsat)
                    # check whether all the strategies have been evaluated
                    if len(computed_utilities_dict) == 2**len(self.decision_atoms_list):
                        print(f"Evaluated all {2**len(self.decision_atoms_list)} possible strategies")
                        break
                    # cleanup - crucial
                    self.asp_program = original_prg.copy()
                
            if self.verbose:
                print(f"Tested {len(computed_utilities_dict)} different strategies")
            if self.pedantic:
                print("Utilities")
                for ut in computed_utilities_dict:
                    print(f"{ut}: {computed_utilities_dict[ut]}")

            # To recycle the method
            best_found = computed_utilities_dict[bin_value_current_strategy]
            computed_utilities_dict = {}
            computed_utilities_dict[bin_value_current_strategy] = best_found
            return self.extract_best_utility(computed_utilities_dict)


    def decision_theory_naive_method(self,
            no_mix : bool = False,
        ) -> 'tuple[tuple[float,float,float],list[str]]':
        '''
        Naive implementation of decision theory: enumerates all the possible
        combinations of decision atoms and then ask the queries (i.e., the
        probability of all the utility atoms) at every iteration.
        Mainly used for reference: easy to implement but not optimal.
        Returns the pair [lower utility, upper utility] for the best strategy.
        The lower utility is:
        sum_{(a,r) in U, r > 0} r * LP{sigma}(a) + sum_{(a,r) in U, r < 0} r * UP{sigma}(a)
        The upper utility is:
        sum_{(a,r) in U, r > 0} r * UP{sigma}(a) + sum_{(a,r) in U, r < 0} r * LP{sigma}(a)
        where LP and UP are respectively the lower and upper probability for the atom a.
        If no_mix is true computes:
        lower utility = sum_{(a,r) in U} r * LP{sigma}(a)
        upper utility = sum_{(a,r) in U} r * UP{sigma}(a)
        This may yield lower utility greater then the upper utility.
        opt true performs maximization over the answer set of every world.
        '''
        decision_facts_combinations = list(range(0, 2**len(self.decision_atoms_list)))
        computed_utilities_list : 'dict[str,list[float]]' = {}

        if len(self.decision_atoms_list) == 0:
            utils.print_error_and_exit("Specify at least one decision atom.")

        original_prg = self.asp_program.copy()
        for el in decision_facts_combinations:
            bin_value = bin(el)[2:].zfill(len(self.decision_atoms_list))
            st = ""
            for index, v in enumerate(bin_value):
                if int(v) == 1:
                    st += f"{self.decision_atoms_list[int(index)]} "
            if self.verbose:
                print(f"----- Strategy: [{st}] -----")
            # s = "0"*(len(self.decision_atoms_list) - len(bin(el)[2:]))
            # add the constraints for the truth values of the facts
            # constraints : list[str] = []
            for index, v in enumerate(bin_value):
                mode = "" if int(v) == 0 else "not"
                c = f":- {mode} {self.decision_atoms_list[index]}."
                # constraints.append(c)
                self.asp_program.append(c)

            # ask the queries
            original_prg_constr = self.asp_program.copy()
            current_utility_l : float = 0
            current_utility_u : float = 0
            for query in self.utilities_dict:
                if self.verbose:
                    print(f"Query: {query}")
                self.asp_program.append(f"q:- {query}.")
                self.asp_program.append("#show q/0.")
                self.asp_program.append(f"nq:- not {query}.")
                self.asp_program.append("#show nq/0.")

                self.compute_probabilities()
                # lp = self.lower_probability_query
                # up = self.upper_probability_query
                self.model_handler = ModelsHandler(
                    self.prob_facts_dict,
                    self.evidence,
                    self.abducibles_list,
                    self.decision_atoms_list,
                    self.utilities_dict
                )
                # self.upper_probability_query = 0
                # self.lower_probability_query = 0
                current_reward = self.utilities_dict[query]
                if no_mix:
                    current_utility_l += self.lower_probability_query * current_reward
                    current_utility_u += self.upper_probability_query * current_reward
                else:
                    if current_reward > 0:
                        # lower util -> lower prob * reward
                        # upper util -> upper prob * reward
                        current_utility_l += self.lower_probability_query * current_reward
                        current_utility_u += self.upper_probability_query * current_reward
                    else:
                        # lower util -> upper prob * reward
                        # upper util -> lower prob * reward
                        current_utility_l += self.upper_probability_query * current_reward
                        current_utility_u += self.lower_probability_query * current_reward
                    
                self.asp_program = original_prg_constr.copy()
            computed_utilities_list[bin_value] = [current_utility_l, current_utility_u]
            if self.verbose:
                print(f"Strategy: {bin_value}, Utility: [{current_utility_l, current_utility_u}]")

            # here, super important: restore the program
            self.asp_program = original_prg.copy()
    
        if self.pedantic:
            print("Utilities")
            for ut in computed_utilities_list:
                print(f"{ut}: {computed_utilities_list[ut]}")
        
        return self.extract_best_utility(computed_utilities_list, self.upper)
 


    def decision_theory_project(self) -> 'tuple[list[float],list[str]]':
        '''
        Decision theory by computing the projective solutions.
        '''
        ctl = self.init_clingo_ctl(["0", "--project"])

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                # print((str(m)))
                self.model_handler.add_decision_model(str(m))  # type: ignore
                self.computed_models = self.computed_models + 1
                # n_models = n_models + 1
            handle.get()  # type: ignore

        selected_strategy, self.utility = self.model_handler.compute_best_strategy()
        for i in range(0,len(selected_strategy)):
            self.decision_atoms_selected.append(self.decision_atoms_list[i] if int(selected_strategy[i]) == 1 else f"not {self.decision_atoms_list[i]}")
            self.decision_atoms_selected
        return self.utility, self.decision_atoms_selected


    def decision_theory_approximate(self,
        initial_population_size : int = 50,
        mutation_probability : float = 0.05,
        samples_for_inference : int = 5000,
        max_iterations_genetic : int = 1000,
        to_maximize : str = "upper"
        ) -> 'tuple[list[float],list[str]]':
        '''
        Implementation of a genetic algorithm algorithm to compute
        the best strategy. The utility of every strategy is approximately
        computed, since we use approximate inference to compute the
        probability of every utility atom.
        https://it.mathworks.com/help/gads/how-the-genetic-algorithm-works.html
        '''
        class Individual:
            '''
            Individual for the population
            '''
            def __init__(self, id_individual : str, score : 'list[float]') -> None:
                self.id_individual = id_individual
                self.score_l = score[0]
                self.score_u = score[1]
            
            def __str__(self) -> str:
                return f"id: {self.id_individual} lower score: {self.score_l} upper score: {self.score_u}"

            def __repr__(self) -> str:
                return self.__str__()
 
    
        def evaluate_score(id_individual: str) -> 'list[float]':
            '''
            Computes the score of an individual by creating the PASP
            with the specified decision atoms and then by computing the 
            probability of every utility fact by sampling.
            '''
            # call self.sample_query() for every utility atom, but before
            # set the query
            # TODO: refactor, copied from decision_theory_naive_method
            computed_utility : 'list[float]' = []
            # print(self.utilities_dict)
            
            # self.print_asp_program()
            original_prg = self.asp_program.copy()
            
            # TEST
            # id = "11"
            
            if self.verbose:
                print(f"Combination: {str(id_individual)}")
            # s = "0"*(len(self.decision_atoms_list) - len(bin(el)[2:]))
            # add the constraints for the truth values of the facts
            # constraints: list[str] = []
            for index, v in enumerate(id_individual):
                mode = "" if int(v) == 0 else "not"
                c = f":- {mode} {self.decision_atoms_list[index]}."
                # constraints.append(c)
                self.asp_program.append(c)

            # ask the queries
            original_prg_constr = self.asp_program.copy()
            current_utility_l: float = 0
            current_utility_u: float = 0
            for query in self.utilities_dict:
                self.asp_program.append(f"q:- {query}.")
                self.asp_program.append("#show q/0.")
                self.asp_program.append(f"nq:- not {query}.")
                self.asp_program.append("#show nq/0.")

                lp, up = self.sample_query()

                self.model_handler = ModelsHandler(
                    self.prob_facts_dict,
                    self.evidence,
                    self.abducibles_list,
                    self.decision_atoms_list,
                    self.utilities_dict
                )

                current_reward = self.utilities_dict[query]
                if current_reward > 0:
                    current_utility_l += lp * current_reward
                    current_utility_u += up * current_reward
                else:
                    current_utility_l += up * current_reward
                    current_utility_u += lp * current_reward
                    
                self.asp_program = original_prg_constr.copy()
            
            computed_utility = [current_utility_l, current_utility_u]
            if self.verbose:
                print(f"Strategy: {id_individual}, Utility: [{current_utility_l, current_utility_u}]")
            self.asp_program = original_prg.copy()

            return computed_utility


        def sample_individual() -> 'Individual':
            '''
            Samples an individual for the population (i.e, a strategy).
            '''
            id_individual : str = ""
            for _ in self.decision_atoms_list:
                if random.random() < 0.5:
                    id_individual += "1"
                else:
                    id_individual += "0"
            return Individual(id_individual, evaluate_score(id_individual))

            
        def init_population(size : int) -> 'list[Individual]':
            '''
            Initializes the population (strategies).
            '''
            pop : 'list[Individual]' = []
            while len(pop) < size:
                ind = sample_individual()
                if ind.id_individual not in [i.id_individual for i in pop]:
                    pop.append(ind)
            return pop

        # self.print_asp_program()
        # print(self.decision_atoms_list)

        if initial_population_size > 2**(len(self.decision_atoms_list)):
            utils.print_warning(f"Initial population size ({initial_population_size}) should be less than the number of possible strategies ({2**len(self.decision_atoms_list)}).")
            utils.print_warning("Setting initial population size to number of possible strategies.")
            initial_population_size = 2**(len(self.decision_atoms_list))

        population : 'list[Individual]' = init_population(initial_population_size)
        if to_maximize == "lower":
            population.sort(key=lambda x : x.score_l, reverse=True)
        else:
            population.sort(key=lambda x: x.score_u, reverse=True)
         
        self.n_samples = samples_for_inference
        
        for it in range(max_iterations_genetic):
            if it % 100 == 0:
                print(f"Iteration {it} - best: {population[0]}")
            best_a = population[0]
            best_b = population[1]
            crossover_position = random.randint(0, len(best_a.id_individual) - 1)
            # combine the two ids
            l_id = list(best_a.id_individual[:crossover_position] + best_b.id_individual[crossover_position:])
            
            # mutation
            for i in range(0,len(l_id)):
                if random.random() < mutation_probability:
                    l_id[i] = '0' if l_id[i] == '1' else '1'
            new_element_id = ''.join(l_id)
            
            # find whether the current element is in the list
            found = False
            for l in population:
                if l.id_individual == new_element_id:
                    found = True
                    break

            if not found:
                score = evaluate_score(new_element_id)
                # ordered insert
                i = 0
                for i in range(0,len(population)):
                    if to_maximize == "lower":
                        if population[i].score_l < score[0]:
                            break
                    else:
                        if population[i].score_u < score[1]:
                            break
                population.insert(i,Individual(new_element_id, score))
                
                # remove the element with the lowest score 
                population = population[:-1]

        # print(population)
        best_comb : 'list[str]' = []
        for c, decision in zip(population[0].id_individual, self.decision_atoms_list):
            if int(c) == 1:
                best_comb.append(decision)
            else:
                best_comb.append(f"not {decision}")

        return [population[0].score_l, population[0].score_u], best_comb


    def abduction_approximate(
        self,
        threshold : float,
        only_smallest_cardinality : bool,
        initial_population_size : int = 50,
        mutation_probability : float = 0.05,
        samples_for_inference : int = 5000,
        max_iterations_genetic : int = 1000,
        target_probability : str = "lower"
        ) -> 'tuple[list[float],list[str]]':
        '''
        Implementation of a genetic algorithm algorithm to compute
        the abductive explanations. The probability of a query is
        approximated via sampling.
        TODO: everithing except evaluate score (which is as well heavily
        copy-pasted) is the same as the
        approximate dt via genetic algorithm.
        https://it.mathworks.com/help/gads/how-the-genetic-algorithm-works.html
        '''
        class Individual:
            '''
            Individual for the population
            '''
            def __init__(self,
                id_individual: str,
                score: 'list[float]'
                ) -> None:
                
                self.id_individual = id_individual
                self.score_l = score[0]
                self.score_u = score[1]
                self.lp = score[2]
                self.up = score[3]

            def __str__(self) -> str:
                return f"id: {self.id_individual}\
                    lower score: {self.score_l}\
                    upper score: {self.score_u}\
                    prob: {self.lp,self.up}".replace("                    "," ")

            def __repr__(self) -> str:
                return self.__str__()

        def evaluate_score(
                id_individual: str,
                only_smallest_cardinality : bool,
                threshold : float = -1
            ) -> 'list[float]':
            '''
            Evaluate the score of an individual: ask the query in the program
            where the abducibles are ste to true or false according to the id
            of the individual.
            '''
            min_score = -1000
            
            original_prg = self.asp_program.copy()

            # TEST
            # id = "11"
            # self.verbose = True
            # id_individual = "10"
            if self.verbose:
                print(f"Combination: {str(id_individual)}")
            # s = "0"*(len(self.decision_atoms_list) - len(bin(el)[2:]))
            # add the constraints for the truth values of the facts
            # constraints: list[str] = []
            for index, v in enumerate(id_individual):
                mode = "" if int(v) == 0 else "not"
                c = f":- {mode} {self.abducibles_list[index]}."
                # constraints.append(c)
                self.asp_program.append(c)
            
            # ask the queries
            try:
                lp, up = self.sample_query()
            except:
                lp = min_score
                up = min_score
                
            self.model_handler = ModelsHandler(
                self.prob_facts_dict,
                self.evidence,
                self.abducibles_list,
                self.decision_atoms_list,
                self.utilities_dict
            )

            if lp == min_score:
                computed_score = [min_score, min_score, lp, up]
            else:
                # score computation: difference between target and threshold
                computed_score = [lp - threshold, up - threshold, lp, up]
            
            if self.verbose:
                print(f"Evaluating score for: {id_individual}, score: {computed_score}")
            self.asp_program = original_prg.copy()
            
            # sys.exit()
            
            return computed_score

        def sample_individual(
            only_smallest_cardinality : bool,
            threshold : float = -1) -> 'Individual':
            '''
            Samples an individual for the population (i.e, a set of abducibles).
            '''
            id_individual : str = ""
            for _ in self.abducibles_list:
                if random.random() < 0.5:
                    id_individual += "1"
                else:
                    id_individual += "0"
            return Individual(id_individual, evaluate_score(id_individual,only_smallest_cardinality,threshold))

        def init_population(size : int, only_smallest_cardinality : bool,threshold : float = -1) -> 'list[Individual]':
            '''
            Initializes the population (list of PASP).
            '''
            pop : 'list[Individual]' = []
            
            while len(pop) < size:
                ind = sample_individual(only_smallest_cardinality,threshold)
                if ind.id_individual not in [i.id_individual for i in pop]:
                    pop.append(ind)
                    if self.verbose:
                        print(f"Inserted element {len(pop)}")
            return pop

        # body of the method
        if initial_population_size > 2**(len(self.abducibles_list)):
            utils.print_warning(f"Initial population size ({initial_population_size}) should be less than the number of possible combinations ({2**len(self.abducibles_list)}) of abducibles.")
            utils.print_warning("Setting initial population size to number of possible combinations of abducibles.")
            initial_population_size = 2**(len(self.abducibles_list))

        # preprocess self.asp_program to make it compatible with sample_query
        l1 = [a for a in self.asp_program if not a.startswith('#show')]
        self.asp_program = l1
        self.asp_program.append("#show q/0.")
        self.asp_program.append("#show nq/0.")
        pfl : list[str] = []
        for pf in self.prob_facts_dict:
            self.asp_program.append(f"#external {pf}.")
            pfl.append('0{' + pf + '}1.')
        self.asp_program = list(set.difference(set(self.asp_program),set(pfl)))
        
        population : 'list[Individual]' = init_population(initial_population_size, only_smallest_cardinality,threshold)
        # how to sort: first criterion, the score, second criterion, the number of abducibles
        if target_probability == "lower":
            # population.sort(key=lambda x : (x.score_l, x.id_individual.count('0')), reverse=True)
            population.sort(key=lambda x : (x.score_l * x.id_individual.count('0')), reverse=True)
        else:
            population.sort(key=lambda x : (x.score_u, x.id_individual.count('0')), reverse=True)
                
        self.n_samples = samples_for_inference

        for it in range(max_iterations_genetic):
            if it % 100 == 0:
                print(f"Iteration {it} - best: {population[0]}")

            best_a = population[0]
            best_b = population[1]
            crossover_position = random.randint(0, len(best_a.id_individual) - 1)
            
            # combine the two ids
            l_id_0 = list(best_a.id_individual[:crossover_position] + best_b.id_individual[crossover_position:])
            l_id_1 = list(best_b.id_individual[:crossover_position] + best_a.id_individual[crossover_position:])

            # mutation
            for i in range(0, len(l_id_0)):
                if random.random() < mutation_probability:
                    l_id_0[i] = '0' if l_id_0[i] == '1' else '1'
            new_element_id_0 = ''.join(l_id_0)
            for i in range(0, len(l_id_1)):
                if random.random() < mutation_probability:
                    l_id_1[i] = '0' if l_id_1[i] == '1' else '1'
            new_element_id_1 = ''.join(l_id_1)
            
            # find whether the current element is in the list
            for new_element_id in [new_element_id_0, new_element_id_1]:
                found = False
                for l in population:
                    if l.id_individual == new_element_id:
                        found = True
                        break

                if not found:
                    score = evaluate_score(new_element_id, only_smallest_cardinality)
                    population.append(Individual(new_element_id, score))
                    
                    if target_probability == "lower":
                        # population.sort(key=lambda x : (x.score_l, x.id_individual.count('0')), reverse=True)
                        population.sort(key=lambda x : (x.score_l * x.id_individual.count('0')), reverse=True)
                    else:
                        population.sort(key=lambda x : (x.score_u, x.id_individual.count('0')), reverse=True)

                    # ordered insert: maybe do this
                    # i = 0
                    # for i in range(0, len(population)):
                    #     if target_probability == "lower":
                    #         if population[i].score_l < score[0]:
                    #             break
                    #     else:
                    #         if population[i].score_u < score[1]:
                    #             break
                    # population.insert(i, Individual(new_element_id, score))

                    # remove the element with the lowest score
                    population = population[:-1]

        best_comb: 'list[str]' = []
        for c, decision in zip(population[0].id_individual, self.decision_atoms_list):
            if int(c) == 1:
                best_comb.append(decision)
            else:
                best_comb.append(f"not {decision}")

        i = 0
        # TODO: since approximate, add tolerance so, score > -0.05 or something
        # along this line
        epsilon = 0.05
        for el in population:
            target_prob = el.lp if target_probability == "lower" else el.up
            if target_prob > threshold - epsilon:
                if self.pedantic:
                    print(f"added {el.id_individual,el.id_individual.count('1'), target_prob}")
                self.abductive_explanations.append(self.model_handler.get_abducibles_from_id(el.id_individual))


    def __abduction_iter(self,
        n_abd: int,
        previously_computed : 'list[str]',
        one_shot : bool = False
        ) -> 'list[str]':
        '''
        Loop for exact abduction.
        If one_shot is true then it does not insert the constraint since we
        call the ASP solver only once to compute all the projected answer sets. 
        '''
        clauses = self.asp_program.copy()
        # for deterministic abduction
        for c in self.cautious_consequences:
            clauses.append(f':- not {c}.')
        if len(self.prob_facts_dict) == 0:
            clauses.append(':- not q.')
        
        if not one_shot:
            clauses.append('abd_facts_counter(C):- #count{X : abd_fact(X)} = C.')
            clauses.append(f':- abd_facts_counter(C), C != {n_abd}.')

        # TODO: instead of, for each iteration, rewriting the whole program,
        # use multi-shot with Number

        ctl = self.init_clingo_ctl(["0", "--project"], clauses)

        for exp in previously_computed:
            s = ":- "
            for el in exp:
                if el != "q" and not el.startswith('not_abd'):
                    s = s + el + ","
            s = s[:-1] + '.'
            ctl.add('base', [], s)

        ctl.ground([("base", [])])

        computed_models : 'list[str]' = []

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                computed_models.append(str(m))  # type: ignore
                # n_models = n_models + 1
            handle.get()  # type: ignore

        return computed_models


    def abduction(self,
        threshold : float = -1,
        only_smallest_cardinality : bool = False,
        one_shot : bool = False
        ) -> None:
        '''
        Abduction. If threshold > 0, constrained abduction find the minimal
        set of facts such taht the probability of the query is above the
        threshold.
        only_smallest_cardinality: if True, find the set of the smallest cardinality.
        If false, the algorithm loops over all the possible solutions.
        If one_shot is true, we compute all the projected answer sets in one shot.
        '''
        if len(self.abducibles_list) == 0:
            utils.print_error_and_exit("Specify at least one abducible.")

        computed_abducibles_list : 'list[str]' = []

        for i in range(0, len(self.abducibles_list) + 1):
            currently_computed = self.__abduction_iter(i, computed_abducibles_list, one_shot)
            self.computed_models = self.computed_models + len(currently_computed)
            if self.verbose:
                if not one_shot:
                    print(f"Models with {i} abducibles: {len(currently_computed)}")
                else:
                    print(f"Total number of models: {len(currently_computed)}")
                if self.pedantic:
                    print(currently_computed)

            # TODO: handle len(currently_computed) > 0 and i == 0 (true without abducibles)

            if len(self.prob_facts_dict) == 0:
                # deterministic abduction
                for i in range(0, len(currently_computed)):
                    currently_computed[i] = currently_computed[i].split(' ')  # type: ignore
                    self.abductive_explanations.append(currently_computed[i])  # type: ignore

                self.computed_models = self.computed_models + len(currently_computed)

                for cc in currently_computed:
                    computed_abducibles_list.append(cc)
            else:
                # if i == 0 and len(currently_computed) != 2**(len(self.prob_facts_dict) - self.n_probabilistic_ics):
                #     utils.print_inconsistent_program(self.stop_if_inconsistent)
                for el in currently_computed:
                    self.model_handler.add_model_abduction(str(el))

            # keep the best model
            self.lower_probability_query, self.upper_probability_query = self.model_handler.keep_best_model(threshold=threshold)
            # this is ok if I search the one with the lowest cardinality

            if self.lower_probability_query >= threshold and threshold != -1 and only_smallest_cardinality:
                break
            
            # if one shot stop the loop
            if one_shot:
                break

        for el in self.model_handler.abd_worlds_dict:
            # if self.stop_if_inconsistent is True and len(self.model_handler.abd_worlds_dict[el].probabilistic_worlds) != 2**len(self.prob_facts_dict):
            consistent = len(self.model_handler.abd_worlds_dict[el].probabilistic_worlds) == 2**len(self.prob_facts_dict)
            if consistent:
                self.abductive_explanations.append(self.model_handler.get_abducibles_from_id(el))
            elif not consistent and self.normalize_prob:
                self.abductive_explanations.append(self.model_handler.get_abducibles_from_id(el))

            # TODO: add normalization, as in compute_probabilities

        if len(self.prob_facts_dict) == 0:
            if len(self.abductive_explanations) > 0:
                self.lower_probability_query = 1
                self.upper_probability_query = 1


    def compute_probability_lpmln(self, query : str) -> None:
        '''
        Computes the probability for a probabilistic answer set program following the
        LPMLN semantics. Most of the data structures are reused,
        so 'probability' should be considered as 'weight' until the
        values are not normalized.
        '''
        ctl = self.init_clingo_ctl(["0"])
        nf : float = 0
        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                nf += self.model_handler.add_value_lpmln(str(m), query)  # type: ignore
                self.computed_models = self.computed_models + 1
            handle.get()   # type: ignore
        self.model_handler.normalize_weights_as(nf)
        self.lower_probability_query, self.upper_probability_query = self.model_handler.compute_lower_upper_probability(self.k_credal)


    def __extract_query_equation(self,
        target : str,
        task : str,
        iter_simplification : bool = False,
        mod_simplification : int = 100
        ) -> str:
        '''
        Computes the query equation.
        task can be 'optimizable' or 'reducible'
        iter_simplification is true if the method simplify iteratively the equation
        mod_simplification: number of iterations after which simplifying the equation
        '''
        from sympy import simplify
        
        pf_as_list = self.prob_facts_dict.items()

        eq = ""
        it = 0
        for k, w in self.model_handler.worlds_dict.items():
            if (target == "upper" and w.model_query_count > 0) or \
                    (target == "lower" and w.model_query_count > 0 and w.model_not_query_count == 0):
                for pf_0_1, pf_data in zip(k, pf_as_list):
                    cleaned_fact = pf_data[0].replace('(', '_').replace(')', '_').replace(',', '_')
                    considered_facts = self.optimizable_facts if task == "optimizable" else self.reducible_facts
                    if f"P({cleaned_fact})" in considered_facts:
                        if pf_0_1 == '0':
                            eq += "(1-" + f"P({cleaned_fact})" + ")"
                        else:
                            eq += f"P({cleaned_fact})"
                    else:
                        if pf_0_1 == '0':
                            eq += str(1 - pf_data[1])[:6]
                        else:
                            eq += str(pf_data[1])[:6]
                    eq += " * "
                eq = eq[:-3]  # remove the last *
                eq += " + "

            if len(eq) > 0 and iter_simplification and (it + 1) % mod_simplification == 0:
                print(f"simplification: {it}/{len(self.model_handler.worlds_dict)}")
                eq = str(simplify(eq[:-2])) + " + "
            it += 1
        
        eq = eq[:-2]  # remove the last +
        if eq == "":
            utils.print_error_and_exit(
                "Error in gathering the query equation.")
        
        return eq


    def reducible_task(self,
        target : str,
        threshold : float,
        simplify_iter : int = -1
        ):
        '''
        Soves the reducible task.
        '''

        if len(self.reducible_facts) == 0:
            utils.print_error_and_exit("Specify at least one reducible fact")
        
        if threshold < 0 or threshold > 1:
            utils.print_error_and_exit(f"The threshold must be between 0 and 1, found {threshold}.")
        
        start_time = time.time()
        if self.verbose:
            print("Computing answer sets")
        self.compute_probabilities()
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"Answer set generation time: {elapsed_time} s")
        
        print(target)
        target = "lower"
        eq = self.__extract_query_equation(target, "reducible", simplify_iter > 0, simplify_iter)
        
        # bu now I assume that there is only one constraint
        # of the form 'query > value'
        # if len(self.constraints) == 0:
        #     utils.print_error_and_exit("Specify at least one constraint.")
        # constraint_on_probability = float(self.constraints[0].split('>')[1])
        start_time = time.time()
        if self.verbose:
            print("Starting optimization")

        found, selected, computed_prob = reduce_pasp_up(
            eq,
            self.reducible_facts,
            threshold
        )
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Optimization time: {elapsed_time} s")
        
        sel = {}
        if found:
            # print("Solution found")
            for name, sl in zip(self.reducible_facts.keys(),selected):
                sel[name] = sl
        
        return found, sel, computed_prob
    

    def optimize_prob(self,
            target : str,
            threshold: float,
            epsilon : float,
            method : str,
            chunk : int
        ):
        '''
        Solves the optimization task.
        '''
        if len(self.optimizable_facts) == 0:
            utils.print_error_and_exit("Specify at least one optimizable fact")
        
        start_time = time.time()
        if self.verbose:
            print("Computing answer sets")
        self.compute_probabilities()
        elapsed_time = time.time() - start_time
        
        if self.verbose:
            print(f"Answer set generation time: {elapsed_time} s")
        
        eq = self.__extract_query_equation(target, "optimizable")
        print(f"number of sums: {eq.count('+')}")
        print(f"number of prods: {eq.count('*')}")
        
        if self.pedantic:
            print(f"Equation: {eq}")
        # bu now I assume that there is only one constraint
        # of the form 'query > value'
        # if len(self.constraints) == 0:
        #     utils.print_error_and_exit("Specify at least one constraint.")
        # constraint_on_probability = float(self.constraints[0].split('>')[1])
        start_time = time.time()
        if self.verbose:
            print("Starting optimization")

        res = compute_optimal_probability(
            eq,
            self.optimizable_facts,
            threshold,
            epsilon,
            method,
            chunk
        )
        elapsed_time = time.time() - start_time
        if self.verbose:
            print(f"Optimization time: {elapsed_time} s")
        
        for el, val in zip(self.optimizable_facts.keys(), res.x):
            eq = eq.replace(el, str(val))

        print(f"Target equation: {eval(eq)}")

        return res

    def print_asp_program(self) -> None:
        '''
        Utility that prints the answer set program.
        '''
        print("--- Answer set program ---")
        print(*self.asp_program, sep='\n')
        for c in self.cautious_consequences:
            print(f":- not {c}.")
        print("---")
