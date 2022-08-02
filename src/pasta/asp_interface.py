import random
import time
import sys

try:
    import clingo
except:
    sys.exit('Install clingo')

from models_handler import ModelsHandler


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
        n_samples : int = 1000
        ) -> None:
        self.cautious_consequences : 'list[str]' = []
        self.program_minimal_set : 'list[str]' = sorted(set(program_minimal_set))
        self.asp_program : 'list[str]' = sorted(set(asp_program))
        self.lower_probability_query : float = 0
        self.upper_probability_query : float = 0
        self.upper_probability_evidence : float = 0
        self.lower_probability_evidence : float = 0
        self.evidence : str = evidence
        self.n_prob_facts : int = len(probabilistic_facts)
        self.abducibles_list : 'list[str]' = abducibles_list
        self.n_abducibles : int = len(self.abducibles_list)
        self.constraint_times_list : 'list[float]' = []
        self.computed_models : int = 0
        self.grounding_time : float = 0
        self.n_worlds : int = 0
        self.world_analysis_time : float = 0
        self.computation_time : float = 0
        self.abductive_explanations : 'list[list[str]]' = []
        self.abduction_time : float = 0
        self.verbose : bool = verbose
        self.pedantic : bool = pedantic
        self.n_samples : int = n_samples
        self.normalizing_factor : float = 0
        self.inconsistent_worlds : 'dict[str,float]' = dict()
        self.prob_facts_dict : 'dict[str,float]' = probabilistic_facts
        self.model_handler : ModelsHandler = \
            ModelsHandler(
                self.prob_facts_dict,
                self.evidence,
                self.abducibles_list)


    def get_minimal_set_facts(self) -> float:
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
        ctl = clingo.Control(["--enum-mode=cautious", "-Wnone"])
        for clause in self.program_minimal_set:
            ctl.add('base',[],clause)

        ctl.ground([("base", [])])
        start_time = time.time()

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

        # sys.exit()
        clingo_time = time.time() - start_time

        return clingo_time

    def compute_probabilities(self, normalize_prob : bool = False, stop_if_inconsistent : bool = False) -> None:
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
        ctl = clingo.Control(["0","--project","-Wnone"])
        for clause in self.asp_program:
            ctl.add('base',[],clause)

        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                ctl.add('base',[],":- not " + c + '.')

        start_time = time.time()
        ctl.ground([("base", [])])
        self.grounding_time = time.time() - start_time

        start_time = time.time()

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                self.model_handler.add_value(str(m))  # type: ignore
                self.computed_models = self.computed_models + 1
            handle.get()   # type: ignore
        self.computation_time = time.time() - start_time

        start_time = time.time()

        if normalize_prob or stop_if_inconsistent:
            ks = sorted(self.model_handler.worlds_dict.keys())
            l : 'list[int]' = []
            for el in ks:
                l.append(int(el,2))

            missing = sorted(set(range(0, 2**self.n_prob_facts)).difference(l))

            if stop_if_inconsistent and len(missing) > 0:
                sys.exit(f"Found worlds without answer sets: {missing}")

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
                print(np)
                self.inconsistent_worlds[n] = np
            if self.verbose:
                print(f"n missing {len(missing)}")
            if self.pedantic:
                print(self.inconsistent_worlds)
            
        self.lower_probability_query, self.upper_probability_query = self.model_handler.compute_lower_upper_probability()

        self.n_worlds = self.model_handler.get_number_worlds()

        self.world_analysis_time = time.time() - start_time


    def sample_world(self):
        '''
        Samples a world for approximate probability computation
        '''
        w_id = ""

        for key in self.prob_facts_dict:
            if random.random() < self.prob_facts_dict[key]:
                w_id = w_id + "T"
            else:
                w_id = w_id + "F"

        return w_id


    def pick_random_index(self, block : int, w_id : str) -> 'list[int]':
        '''
        Pick a random index, used in Gibbs sampling.
        TODO: this can be a static method.
        '''
        # i = random.randint(0,len(id) - 1)
        # while i == 1:
        # 	i = random.randint(0,len(id) - 1)
        return sorted(set([random.randint(0, len(w_id) - 1) for _ in range(0, block)]))

    def resample(self, i : int) -> str:
        '''
        Resamples a facts. Used in MH sampling.
        '''
        key : str = ""
        for k in self.prob_facts_dict:
            key = k
            i = i - 1
            if i < 0:
                break

        if random.random() < self.prob_facts_dict[key]:
            return 'T'
        else:
            return 'F'


    @staticmethod
    def compute_conditional_lp_up(
        n_lower_qe : int,
        n_upper_qe : int,
        n_lower_nqe : int,
        n_upper_nqe : int,
        n_samples : int
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

        lp = lower_q_e / (lower_q_e + upper_not_q_e) if (lower_q_e + upper_not_q_e) > 0 else 0
        up = upper_q_e / (upper_q_e + lower_not_q_e) if (upper_q_e + lower_not_q_e) > 0 else 0

        return lp, up


    @staticmethod
    def assign_T_F_and_get_count(ctl : clingo.Control, w_id : str) -> 'tuple[int,int,int,int]':
        '''
        It does what it is specified in its name.
        '''
        i = 0
        for atm in ctl.symbolic_atoms:
            if atm.is_external:
                ctl.assign_external(atm.literal, w_id[i] == 'T')
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
    def assign_T_F_and_check_if_evidence(ctl : clingo.Control, w_id : str) -> bool:
        '''
        Assigns T or F to facts and checks whether q and e or not q and e
        is true.
        Used in Gibbs sampling
        '''
        i : int = 0

        for atm in ctl.symbolic_atoms:
            if atm.is_external:
                ctl.assign_external(atm.literal, w_id[i] == 'T')
                i = i + 1

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
        w_id : str
        ) -> 'tuple[int,int,int,int]':
        '''
        If the world is has been already considered, retrieve it; otherwise
        compute its contribution.
        Used for sampling
        '''
        if w_id in sampled:
            return sampled[w_id][0], sampled[w_id][1], sampled[w_id][2], sampled[w_id][3]

        qe_count, qe_false_count, nqe_count, nqe_false_count = AspInterface.assign_T_F_and_get_count(ctl, w_id)

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


    def init_clingo_ctl(self) -> 'clingo.Control':
        '''
        Init clingo and grounds the program
        '''
        ctl = clingo.Control(["0", "--project"])
        for clause in self.asp_program:
            ctl.add('base', [], clause)
        ctl.ground([("base", [])])

        return ctl


    def mh_sampling(self) -> 'tuple[float, float]':
        '''
        MH sampling
        '''
        # each element has the structure
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, T_count]
        sampled = {}

        ctl = self.init_clingo_ctl()

        n_samples = self.n_samples

        w_id = self.sample_world()
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
            w_id = self.sample_world()
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
                qe_count, qe_false_count, nqe_count, nqe_false_count = AspInterface.assign_T_F_and_get_count(ctl, w_id)

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

        return AspInterface.compute_conditional_lp_up(n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, n_samples)


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

        ctl = self.init_clingo_ctl()

        n_samples = self.n_samples

        n_lower_qe : int = 0
        n_upper_qe : int = 0
        n_lower_nqe : int = 0
        n_upper_nqe : int = 0

        k : int = 0

        ev : bool = False

        w_id : str = ""
        idNew : str = ""

        while k < n_samples:
            k = k + 1

            # Step 0: sample evidence
            ev = False
            while ev is False:
                w_id = self.sample_world()
                if w_id in sampled_evidence:
                    ev = sampled_evidence[id]
                else:
                    ev = AspInterface.assign_T_F_and_check_if_evidence(ctl, w_id)
                    sampled_evidence[w_id] = ev

            # Step 1: switch samples but keep the evidence true
            ev = False

            while ev is False:
                # blocked gibbs
                to_resample = self.pick_random_index(block, w_id)
                idNew = w_id
                for i in to_resample:
                    idNew = idNew[:i] + self.resample(i) + idNew[i + 1:]

                if idNew in sampled_evidence:
                    ev = sampled_evidence[idNew]
                else:
                    ev = AspInterface.assign_T_F_and_check_if_evidence(ctl, idNew)
                    sampled_evidence[idNew] = ev

            # step 2: ask query
            lower_qe, upper_qe, lower_nqe, upper_nqe = AspInterface.get_val_or_compute_and_update_dict(sampled_query, ctl, idNew)

            n_lower_qe = n_lower_qe + lower_qe
            n_upper_qe = n_upper_qe + upper_qe
            n_lower_nqe = n_lower_nqe + lower_nqe
            n_upper_nqe = n_upper_nqe + upper_nqe

        return AspInterface.compute_conditional_lp_up(n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, n_samples)


    def rejection_sampling(self) -> 'tuple[float, float]':
        '''
        Rejection Sampling
        '''
        # each element has the structure
        # [n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe]
        sampled = {}

        ctl = self.init_clingo_ctl()

        n_lower_qe : int = 0
        n_upper_qe : int = 0
        n_lower_nqe : int = 0
        n_upper_nqe : int = 0

        k : int = 0

        while k < self.n_samples:
            w_id = self.sample_world()
            k = k + 1
            lower_qe, upper_qe, lower_nqe, upper_nqe = AspInterface.get_val_or_compute_and_update_dict(sampled, ctl, w_id)

            n_lower_qe = n_lower_qe + lower_qe
            n_upper_qe = n_upper_qe + upper_qe
            n_lower_nqe = n_lower_nqe + lower_nqe
            n_upper_nqe = n_upper_nqe + upper_nqe

        return AspInterface.compute_conditional_lp_up(n_lower_qe, n_upper_qe, n_lower_nqe, n_upper_nqe, self.n_samples)


    def sample_query(self) -> 'tuple[float, float]':
        '''
        Samples the query self.n_samples times
        If bound is True, stops when either the number of samples taken k
        is greater than self.n_samples or
        2 * 1.96 * math.sqrt(p * (1-p) / k) < 0.02
        '''
        # sampled worlds
        # each element is a list [lower, upper]
        sampled = {}

        ctl = self.init_clingo_ctl()

        n_lower : int = 0
        n_upper : int = 0
        k : int = 0

        while k < self.n_samples:
            w_id = self.sample_world()
            k = k + 1

            if w_id in sampled:
                n_lower = n_lower + sampled[w_id][0]
                n_upper = n_upper + sampled[w_id][1]
            else:
                i = 0
                for atm in ctl.symbolic_atoms:
                    # atm.symbol.name # functor
                    # atm.symbol.arguments[0].number # index
                    if atm.is_external:
                        # possible since dicts are ordered in Python 3.7+
                        ctl.assign_external(atm.literal, w_id[i] == 'T')
                        i = i + 1

                upper_count = 0
                lower_count = 0
                with ctl.solve(yield_=True) as handle:  # type: ignore
                    for m in handle:  # type: ignore
                        m1 = str(m)  # type: ignore
                        if m1 == "q":
                            upper_count = upper_count + 1
                        else:
                            lower_count = lower_count + 1

                        handle.get()  # type: ignore

                up = 1 if upper_count > 0 else 0
                lp = 1 if up and lower_count == 0 else 0

                sampled[w_id] = [lp, up]

                n_lower = n_lower + lp
                n_upper = n_upper + up


            # if bound is True:
            # 	p = n_lower / k
            # 	# condition = 2 * 1.96 * math.sqrt(p * (1-p) / k) >= 0.02
            # 	condition = 2 * 1.96 * math.sqrt(p * (1-p) / k) < 0.02
            # 	if condition and n_lower > 5 and k - n_lower > 5 and k % 101 == 0:
            # 		a = 2 * 1.96 * math.sqrt(p * (1-p) / k)
            # 		break

        return n_lower / k, n_upper / k


    def abduction_iter(self, n_abd: int, previously_computed : 'list[str]') -> 'tuple[list[str], float]':
        '''
        Loop for exact abduction
        '''
        if self.verbose:
            print(str(n_abd) + " abd")

        ctl = clingo.Control(["0", "--project"])
        for clause in self.asp_program:
            ctl.add('base', [], clause)

        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                ctl.add('base', [], ":- not " + c + '.')

        if self.n_prob_facts == 0:
            ctl.add('base', [], ':- not q.')
        ctl.add('base', [], 'abd_facts_counter(C):- #count{X : abd_fact(X)} = C.')
        ctl.add('base', [], ':- abd_facts_counter(C), C != ' + str(n_abd) + '.')
        # TODO: instead of, for each iteration, rewrite the whole program,
        # use multi-shot with Number

        for exp in previously_computed:
            s = ":- "
            for el in exp:
                if el != "q" and not el.startswith('not_abd'):
                    s = s + el + ","
            s = s[:-1] + '.'
            ctl.add('base', [], s)

        start_time = time.time()
        ctl.ground([("base", [])])
        self.grounding_time = time.time() - start_time

        computed_models : 'list[str]' = []

        with ctl.solve(yield_=True) as handle:  # type: ignore
            for m in handle:  # type: ignore
                computed_models.append(str(m))  # type: ignore
                # n_models = n_models + 1
            handle.get()  # type: ignore

        computation_time = time.time() - start_time

        if self.verbose:
            print(f"Time: {computation_time}")

        return computed_models, computation_time


    def abduction(self) -> None:
        '''
        Abduction
        '''
        computed_abducibles_list : 'list[str]' = []

        start_time = time.time()

        for i in range(0, self.n_abducibles + 1):
            currently_computed, exec_time = self.abduction_iter(i, computed_abducibles_list)
            self.computed_models = self.computed_models + len(currently_computed)
            if self.verbose:
                print(f"Models with {i} abducibles: {len(currently_computed)}")
                if self.pedantic:
                    print(currently_computed)

            # TODO: gestire len(currently_computed) > 0 and i == 0 (vero senza abducibili)

            if self.n_prob_facts == 0:
                # currently computed: list of computed models
                for i in range(0, len(currently_computed)):
                    currently_computed[i] = currently_computed[i].split(' ')  # type: ignore
                    self.abductive_explanations.append(currently_computed[i])  # type: ignore

                self.computed_models = self.computed_models + len(currently_computed)

                for cc in currently_computed:
                    computed_abducibles_list.append(cc)
            else:
                for el in currently_computed:
                    self.model_handler.add_model_abduction(str(el))

                self.lower_probability_query, self.upper_probability_query = self.model_handler.compute_lower_upper_probability()

            # keep the best model
            self.lower_probability_query, self.upper_probability_query = self.model_handler.keep_best_model()
            self.constraint_times_list.append(exec_time)

        for el in self.model_handler.abd_worlds_dict:
            self.abductive_explanations.append(self.model_handler.get_abducibles_from_id(el))

        self.abduction_time = time.time() - start_time


    def log_infos(self) -> None:
        '''
        Log some execution details
        '''
        print(f"Computed models: {self.computed_models}")
        print(f"Considered worlds: {self.n_worlds}")
        print(f"Grounding time (s): {self.grounding_time}")
        print(f"Probability computation time (s): {self.computation_time}")
        print(f"World analysis time (s): {self.world_analysis_time}")


    def print_asp_program(self) -> None:
        '''
        Utility that prints the ASP program
        '''
        for el in self.asp_program:
            print(el)
        if len(self.cautious_consequences) != 0:
            for c in self.cautious_consequences:
                print(f":- not {c}.")
