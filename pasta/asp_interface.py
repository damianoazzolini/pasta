""" Module implementing the connection to clingo """

import random

import utils

try:
    import clingo
except:
    utils.print_error_and_exit('Install clingo')

from models_handler import ModelsHandler


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
        k_credal: int = 100
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
        self.normalizing_factor : float = 0
        self.xor: bool = xor
        self.decision_atoms_selected : 'list[str]' = []
        self.utility : float = 0
        self.decision_atoms_list: 'list[str]' = decision_atoms_list
        self.utilities_dict : 'dict[str,float]' = utilities_dict
        self.upper : bool = upper
        self.n_probabilistic_ics : int = n_probabilistic_ics
        self.k_credal : int = k_credal
        self.model_handler : ModelsHandler = \
            ModelsHandler(
                self.prob_facts_dict,
                self.evidence,
                self.abducibles_list,
                self.decision_atoms_list,
                self.utilities_dict
            )
            
    
    def admits_inconsistency(self) -> bool:
        # Currently not used
        return True in {not self.stop_if_inconsistent, self.xor, self.normalize_prob, self.upper, len(self.cautious_consequences) > 0}


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
        if self.k_credal == 100:
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

        ks = sorted(self.model_handler.worlds_dict.keys())
        l : 'list[int]' = []
        for el in ks:
            if el != '':
                l.append(int(el,2))

        missing = sorted(set(range(0, 2**len(self.prob_facts_dict))).difference(l), key=lambda x: bin(x)[2:].count('1'))

        if len(ks) == 0 and len(self.prob_facts_dict) > 0:
            utils.print_inconsistent_program(True)

        ntw = len(self.model_handler.worlds_dict) + 2**(len(self.prob_facts_dict) - len(self.cautious_consequences))
        nw = 2**len(self.prob_facts_dict)

        if len(self.cautious_consequences) > 0 and (ntw != nw) and not self.xor and not self.upper:
            utils.print_inconsistent_program(self.stop_if_inconsistent)
            
        if self.stop_if_inconsistent and len(missing) > 0 and not self.normalize_prob and len(self.prob_facts_dict) > 0:
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
            utils.print_error_and_exit(f"found {len(missing)} worlds without answer sets: {missing}\n{res[:-1]}")

        if self.normalize_prob:
            for el in missing:
                n = str(bin(el))[2:]
                n = str(n).zfill(len(ks[0]))
                i = 0
                np = 1
                for pf in self.prob_facts_dict:
                    if n[i] == '0':
                        np = np * (1 - self.prob_facts_dict[pf])
                    else:
                        np = np * self.prob_facts_dict[pf]
                    i = i + 1
                self.normalizing_factor = self.normalizing_factor + np
                self.inconsistent_worlds[n] = np
            if self.verbose:
                print(f"n missing {len(missing)}")
            if self.pedantic:
                print(self.inconsistent_worlds)

        if self.pedantic:
            print(utils.RED + "lp" + utils.END + utils.YELLOW + " up" + utils.END)
            for el in self.prob_facts_dict:
                print(el, end="\t")
            print("LP/UP\tProbability")
            lp_count = 0
            up_count = 0
            for el in self.model_handler.worlds_dict:
                for i in range(0,len(el)):
                    print(f"{el[i]}", end="\t")
                if self.model_handler.worlds_dict[el].model_query_count > 0 and self.model_handler.worlds_dict[el].model_not_query_count == 0:
                    print(utils.RED + "LP\t", end = "")
                    lp_count = lp_count + 1
                elif self.model_handler.worlds_dict[el].model_query_count > 0 and self.model_handler.worlds_dict[el].model_not_query_count > 0:
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


    def sample_world(self) -> 'tuple[dict[str,bool],str]':
        '''
        Samples a world for approximate probability computation.
        '''
        w_id: 'dict[str,bool]' = {}
        w_id_key: str = ""

        for key in self.prob_facts_dict:
            if random.random() < self.prob_facts_dict[key]:
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


    def mh_sampling(self) -> 'tuple[float, float]':
        '''
        MH sampling
        '''
        # each element has the structure
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, T_count]
        sampled = {}

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
        sampled_evidence = {}
        # list of samples for the query
        # each element has the structure
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe]
        sampled_query = {}

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
                    ev = sampled_evidence[id]
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


    def rejection_sampling(self) -> 'tuple[float, float]':
        '''
        Rejection Sampling
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


    def sample_query(self) -> 'tuple[float, float]':
        '''
        Samples the query self.n_samples times.
        '''
        # sampled worlds
        # each element is a list [lower, upper]
        sampled = {}

        ctl = self.init_clingo_ctl(["0", "--project"])

        n_lower : int = 0
        n_upper : int = 0

        # for _ in utils.progressbar(range(self.n_samples), "Computing: ", 40):
        for _ in range(self.n_samples):
            w_assignments, w_id = self.sample_world()

            if w_id in sampled:
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

                if lower_count == 0 and upper_count == 0:
                    utils.print_inconsistent_program(self.stop_if_inconsistent)

                up = 1 if upper_count > 0 else 0
                lp = 1 if up and lower_count == 0 else 0

                sampled[w_id] = [lp, up]

                n_lower = n_lower + lp
                n_upper = n_upper + up

        return n_lower / self.n_samples, n_upper / self.n_samples
    
    
    def extract_best_utility(self, computed_utilities_list : 'dict[str,list[float]]', lower : bool = False) -> 'tuple[float,list[str]]':
        '''
        Loops over the utility list and find the best assignment.
        '''
        best = -1000
        best_comb : 'list[str]' = []
        for el in computed_utilities_list:    
            index = 0 if lower else 1
            if computed_utilities_list[el][index] > best:
                best = computed_utilities_list[el][index]
                best_comb = []
                for c, decision in zip(el,self.decision_atoms_list):
                    if int(c) == 1:
                        best_comb.append(decision)
                    else:
                        best_comb.append(f"not {decision}")
    
        return best, best_comb


    def decision_theory_naive_method(self) -> 'tuple[float,list[str]]':
        '''
        Naive implementation of decision theory: enumerates all the possible
        combinations of decision atoms and then ask the queries (i.e., the
        probability of all the utility atoms) at every iteration.
        Mainly used for reference: easy to implement but not optimal.
        '''
        decision_facts_combinations = list(range(0, 2**len(self.decision_atoms_list)))
        computed_utilities_list : 'dict[str,list[float]]' = {}

        original_prg = self.asp_program.copy()
        for el in decision_facts_combinations:
            bin_value = bin(el)[2:].zfill(len(self.decision_atoms_list))
            # s = "0"*(len(self.decision_atoms_list) - len(bin(el)[2:]))
            # add the constraints for the truth values of the facts
            constraints : list[str] = []
            for index, v in enumerate(bin_value):
                mode = "" if int(v) == 0 else "not"
                c = f":- {mode} {self.decision_atoms_list[index]}."
                constraints.append(c)
                self.asp_program.append(c)
            
            # ask the queries
            original_prg_constr = self.asp_program.copy()
            current_utility_l : float = 0
            current_utility_u : float = 0
            for query in self.utilities_dict:
                self.asp_program.append(f"q:- {query}.")
                self.asp_program.append("#show q/0.")
                self.asp_program.append(f"nq:- not {query}.")
                self.asp_program.append("#show nq/0.")

                self.compute_probabilities()
                lp = self.lower_probability_query
                up = self.upper_probability_query
                self.model_handler = ModelsHandler(
                    self.prob_facts_dict,
                    self.evidence,
                    self.abducibles_list,
                    self.decision_atoms_list,
                    self.utilities_dict
                )
                self.upper_probability_query = 0
                self.lower_probability_query = 0
                current_utility_l += lp * self.utilities_dict[query]
                current_utility_u += up *self.utilities_dict[query]
                self.asp_program = original_prg_constr.copy()
            computed_utilities_list[bin_value] = [
                current_utility_l, current_utility_u]
            self.asp_program = original_prg.copy()
        
        return self.extract_best_utility(computed_utilities_list)


    def decision_theory(self) -> 'tuple[float,list[str]]':
        '''
        Decision theory naive solver: considers all the possible combinations
        of utility facts
        '''
        ctl = self.init_clingo_ctl(["0", "--project"])

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                self.model_handler.add_decision_model(str(m))  # type: ignore
                self.computed_models = self.computed_models + 1
                # n_models = n_models + 1
            handle.get()  # type: ignore

        self.lower_probability_query, self.upper_probability_query, self.decision_atoms_selected, self.utility = self.model_handler.compute_utility_atoms()
        

    def abduction_iter(self, n_abd: int, previously_computed : 'list[str]') -> 'list[str]':
        '''
        Loop for exact abduction
        '''
        if self.verbose:
            print(str(n_abd) + " abd")

        clauses = self.asp_program.copy()
        for c in self.cautious_consequences:
            clauses.append(f':- not {c}.')
        if len(self.prob_facts_dict) == 0:
            clauses.append(':- not q.')
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


    def abduction(self) -> None:
        '''
        Abduction
        '''
        if len(self.abducibles_list) == 0:
            utils.print_error_and_exit("Specify at least one abducible.")

        computed_abducibles_list : 'list[str]' = []

        for i in range(0, len(self.abducibles_list) + 1):
            currently_computed = self.abduction_iter(i, computed_abducibles_list)
            self.computed_models = self.computed_models + len(currently_computed)
            if self.verbose:
                print(f"Models with {i} abducibles: {len(currently_computed)}")
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
                if i == 0 and len(currently_computed) != 2**(len(self.prob_facts_dict) - self.n_probabilistic_ics):
                    utils.print_inconsistent_program(self.stop_if_inconsistent)
                for el in currently_computed:
                    self.model_handler.add_model_abduction(str(el))

            # keep the best model
            self.lower_probability_query, self.upper_probability_query = self.model_handler.keep_best_model()

        n_inconsistent = 0
        for el in self.model_handler.abd_worlds_dict:
            if self.stop_if_inconsistent is True and len(self.model_handler.abd_worlds_dict[el].probabilistic_worlds) != 2**len(self.prob_facts_dict):
                n_inconsistent = n_inconsistent + 1
            self.abductive_explanations.append(self.model_handler.get_abducibles_from_id(el))
            # TODO: add normalization, as in compute_probabilities

        if len(self.prob_facts_dict) == 0:
            if len(self.abductive_explanations) > 0:
                self.lower_probability_query = 1
                self.upper_probability_query = 1
                

    def compute_probability_lpmln(self, query : str) -> None:
        '''
        Computes the probability for a PASP program following the
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


    def print_asp_program(self) -> None:
        '''
        Utility that prints the ASP program
        '''
        print(*self.asp_program, sep='\n')
        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                print(f":- not {c}.")
