'''
Class defining a parser for a PASTA program.
'''
from io import TextIOWrapper
import os
import re

from utils import print_error_and_exit, error_prob_fact_twice, is_number
from generator import Generator


def distr_to_list(s: str) -> 'list[str | list[str]]':
    '''
    Converts a string such as "gaussian(0,1)." into ["gaussian",["0","1"]].
    Supposes that the string is well-formed.
    '''
    l : 'list[str | list[str]]' = []
    distr = s.split('(')
    l.append(distr[0])
    l.append(distr[1].split(')')[0].split(','))

    return l


def valid_distributions() -> 'list[str]':
    return ["gaussian","uniform","exponential"]


def continuous_distribution_in_str(line : str) -> bool:
    l = line.replace(' ','')
    for d in valid_distributions():
        if ':' + d in l:
            return True
    return False


def symbol_endline_or_space(char1: str) -> bool:
    return char1 == '\n' or char1 == '\r' or char1 == '\r\n' or char1 == '\n\r' or char1 == ' '


def endline_symbol(char1: str) -> bool:
    return char1 == '\n' or char1 == '\r\n' or char1 == '\n\r'


def check_consistent_prob_fact(line_in: str) -> 'tuple[float, str]':
    r = "0.[0-9]+::[a-z_][a-z_0-9]*(\([a-z_A-Z0-9]*(,[a-z_A-Z0-9]*)*\))*\."
    x = re.match(r, line_in.strip())
    if x is None:
        print_error_and_exit(f"Probabilistic fact ->{line_in}<- ill formed")

    line = line_in.split("::")

    return float(line[0]), line[1][:-1]


def get_functor(term: str) -> str:
    '''
    Extracts the functor from a compound term. If the term is an atom
    returns the atom itself.
    '''
    r = ""
    i = 0
    while i < len(term) and term[i] != '(':
        r = r + term[i]
        i = i + 1
    return r


def get_fact_and_utility(term: str) -> 'tuple[str,float]':
    '''
    Extracts the utility and the term from utility(term,utility).
    '''
    t = term.split("utility")[1][1:-2] # eat ). and the initial (
    i = len(t) - 1
    while t[i] != ',' and i > 0:
        i = i - 1
    return t[0:i], float(t[i+1:])


def remove_trailing_comments(line: str) -> str:
    percent = line.find('%')
    if percent != -1:
        line = line[:percent]
    return line


class PastaParser:
    '''
    Parameters:
        - filename: name of the file to read
        - query: query
        - evidence: evidence
        - lines_original: lines from the parsing of the original file
        - lines_prob: lines obtained by parsing probabilistic facts
        - probabilistic_fact: dictionary containing pairs [probabilistic fact, probability]
        - abducibles: list of abducibles
    '''

    def __init__(
        self, 
        filename : str, 
        query : str = "", 
        evidence : str = "",
        for_asp_solver : bool = False,
        naive_dt : bool = False
        ) -> None:
        self.filename : str = filename
        self.query : str = query
        self.evidence : str = evidence
        self.lines_original : 'list[str]' = []
        self.lines_prob : 'list[str]' = []
        self.probabilistic_facts : 'dict[str,float]' = {} # pairs [fact,prob]
        self.continuous_vars : 'dict[str,list[str|list[str]]]' = {}
        self.abducibles : 'list[str]' = []
        self.n_probabilistic_ics : int = 0
        self.body_probabilistic_ics : 'list[str]' = []
        self.map_id_list : 'list[int]' = []
        self.constraints_list : 'list[str]' = []
        self.fact_utility : 'dict[str,float]' = {}
        self.decision_facts : 'list[str]' = []
        self.for_asp_solver : bool = for_asp_solver
        self.naive_dt : bool = naive_dt


    def get_file_handler(self, from_string : str = "") -> TextIOWrapper:
        if not from_string:
            if not os.path.isfile(self.filename):
                print_error_and_exit(f"File {self.filename} not found")
            return open(self.filename, "r")
        else:
            import io
            return io.StringIO(from_string)


    def parse_approx(self, from_string : str = "") -> None:
        '''
        Parses a program into an alternative form: probabilistic 
        facts are converted into external facts
        '''
        f = self.get_file_handler(from_string)
        lines = f.readlines()
        gen = Generator()

        for l in lines:
            l = l.rstrip().lstrip()
            if not l.startswith('%'):
                if '::' in l:
                    # probabilistic fact
                    l = l.replace('\n','').replace('\t','').split('::')
                    prob = l[0]
                    term = l[1].replace('\n','').replace('\r','').replace('.','')
                    self.probabilistic_facts[term] = float(prob)
                    self.lines_prob.append(f'#external {term}.')
                elif l.startswith("("):
                    l = l.replace('\n','').replace('\r','')
                    expanded_conditional = gen.generate_clauses_for_conditionals(l)
                    for el in expanded_conditional:
                        self.lines_prob.append(el)
                elif ':' in l and continuous_distribution_in_str(l):
                    # continuous variable
                    # TODO: add support range a(1..n)
                    l1 = l.split(':')
                    atom = l1[0]
                    distr = l1[1]
                    if '..' in atom:
                        b = atom.split('(')[1].replace(')','').split('..')
                        lb = int(b[0])
                        ub = int(b[1].replace(')', '').replace('.', ''))
                        name = atom.split('(')[0]
                        for i in range(lb, ub + 1):
                            self.continuous_vars[f"{name}({i})"] = distr_to_list(distr)
                    else:
                        self.continuous_vars[atom] = distr_to_list(distr)
                elif '#constraint(' in l:
                    # clause with constraint in the body
                    new_clause : 'list[str]' = []
                    l = l.replace(' ','')
                    # find all the starting positions of #constraint
                    constr = [m.start() for m in re.finditer('#constraint', l)]
                    end_pos : 'list[int]' = []
                    # loop to find the ending positions
                    for c in constr:
                        sc = ""
                        # 12 is len('#constraint') + 1
                        i = c + 12
                        count = 1
                        while count != 0:
                            if l[i] == ')':
                                count = count - 1
                            elif l[i] == '(':
                                count = count + 1
                            sc = sc + l[i]
                            i = i + 1
                        
                        end_pos.append(i)
                        self.lines_prob.append(f"#external constraint_{len(self.constraints_list)}.")
                        new_clause.append(f"constraint_{len(self.constraints_list)}")
                        self.constraints_list.append(sc[:-1])

                    i = 0
                    new_clause.append(l[0:constr[0]])
                    for i in range(1, len(constr)):
                        el = l[end_pos[i - 1]:constr[i]]
                        if el.startswith(','):
                            el = el[1:]
                        if el.endswith(','):
                            el = el[:-1]

                        new_clause.insert(0, el)
                    
                    cl: str = new_clause[-1]
                    for v in new_clause[:-1]:
                        cl = cl + v + ','
                    cl = cl[:-1] + '.'
                    self.lines_prob.append(cl)
                elif not l.startswith('\n'):
                    self.lines_prob.append(l.replace('\n','').replace('\r',''))


    def parse(self, from_string : str = "") -> None:
        '''
        Parses the file
        '''
        f = self.get_file_handler(from_string)
        lines = f.readlines()
        f.close()
        
        i = 0
        while i < len(lines):
            l1 : str = ""
            line = lines[i].replace('\n','').replace('\r','')
            if not line.lstrip().startswith('%') and len(line) > 0:
                ls = line.split('. ')
                if len(ls) > 1:
                    # two probabilistic facts on the same line
                    for ii in range(0,len(ls)):
                        if ii != len(ls) - 1:
                            ls[ii] += '.'
                        self.lines_original.append(ls[ii])
                else:
                    if not line.rstrip().endswith('.'):
                        while not line.rstrip().endswith('.') and i < len(lines):
                            percent = line.find('%')
                            if percent != -1:
                                line = line[:percent]
                            l1 += line
                            i = i + 1
                            line = lines[i].replace('\n', '').replace('\r', '')
                        percent = line.find('%')
                        if percent != -1:
                            line = line[:percent]
                        l1 += line
                        i = i + 1
                    else:
                        l1 = line
                self.lines_original.append(l1)
            i = i + 1

        self.parse_program()


    def parse_program(self) -> bool:
        '''
        Second layer of program parsing: generates the ASP encoding
        for the probabilistic, abducible, map, ... facts
        '''
        n_probabilistic_facts = 0
        gen = Generator()
        for line in self.lines_original:
            self.check_reserved(line)
            if "::" in line and not line.startswith('%') and not line.startswith("map"):
                if ':-' in line:
                    print_error_and_exit("Probabilistic clauses are not supported\n" + line)
                if ';' in line:
                    print_error_and_exit(
                        "Disjunction is not yet supported in probabilistic facts\nplease rewrite it as single fact.\nExample: 0.6::a;0.2::b. can be written as\n0.6::a. 0.5::b. where 0.5=0.2/(1 - 0.6)")
                # line with probability value
                probability, fact = check_consistent_prob_fact(line.replace(' ',''))

                self.add_probabilistic_fact(fact,probability)

                # self.lines_prob.append(clauses)

                n_probabilistic_facts = n_probabilistic_facts + 1
            elif line.startswith("query("):
                # remove the "query" functor and handles whether the line
                # does not terminate with .
                # query(fly(1)) -> fly(1)
                if line[-1] == ".":
                    self.query = line.split("query")[1][:-2][1:]
                else:
                    self.query = line.split("query")[1][:-1][1:]
            elif line.startswith("evidence("):
                if line[-1] == ".":
                    # remove the "evidence" functor and handles whether the line
                    # does not terminate with .
                    # evidence(fly(1)) -> fly(1)
                    self.evidence = line.split("evidence")[1][:-2][1:]
                else:
                    self.evidence = line.split("evidence")[1][:-1][1:]
            elif line.startswith("("):
                expanded_conditional = gen.generate_clauses_for_conditionals(line)
                for el in expanded_conditional:
                    self.lines_prob.append(el)
            elif line.startswith("abducible"):
                _, abducible = gen.generate_clauses_for_abducibles(line, 0)
                # self.lines_prob.append(clauses)
                # self.abducibles.append(abducible)
                self.abducibles.append(abducible)
            elif line.startswith("map"):
                # add the MAP fact as probabilistic
                fact = line.split('map')[1]
                probability, fact = check_consistent_prob_fact(fact)
                self.map_id_list.append(len(self.probabilistic_facts))
                self.add_probabilistic_fact(fact,probability)
            elif line.startswith("decision"):
                fact = line.split('decision')[1][:-1].strip()
                clauses = gen.generate_clauses_for_dt(fact, "decision", self.naive_dt)
                self.decision_facts.append(fact)
                for c in clauses:
                    self.lines_prob.append(c)
            elif line.startswith("utility"):
                fact, utility = get_fact_and_utility(line)
                self.fact_utility[fact] = utility
                # print(f"utility({fact},{int(utility)}):- {fact}.")
                # keep it to possibly impose ASP constraints
                # on the utilites (e.g. on weights?) 
                self.lines_prob.append(line)
                clauses = gen.generate_clauses_for_dt(fact, "utility", self.naive_dt)
                # self.decision_facts.append(fact)
                for c in clauses:
                    self.lines_prob.append(c)

            elif is_number(line.split(':-')[0]):
                # probabilistic IC p:- body.
                # print("prob ic")
                # generate the probabilistic fact
                new_line = line.split(':-')[0] + "::icf" + str(self.n_probabilistic_ics) + "."
                probability, fact = check_consistent_prob_fact(new_line)
                self.add_probabilistic_fact(fact, probability)
                new_clause = "ic" + str(self.n_probabilistic_ics) + ":- " + line.split(':-')[1]
                self.lines_prob.append(new_clause)

                new_ic_0 = ":- icf" + str(self.n_probabilistic_ics) + ", ic" + str(self.n_probabilistic_ics) + "."
                self.lines_prob.append(new_ic_0)

                new_ic_1 = ":- not icf" + str(self.n_probabilistic_ics) + ", not ic" + str(self.n_probabilistic_ics) + "."
                self.lines_prob.append(new_ic_1)

                self.n_probabilistic_ics = self.n_probabilistic_ics + 1
                
            else:
                if not line.startswith("#show"):
                    self.lines_prob.append(line)
        
        if not self.query and len(self.decision_facts) == 0:
            print_error_and_exit("Missing query")

        i = 0
        for fact in self.probabilistic_facts:
            # To handle 0.1::a. a. q:- a.
            # Without this, the computed prob is 0.1, while the correct
            # prob should be 1.
            if fact + '.' in self.lines_prob:
                self.probabilistic_facts[fact] = 1

            if self.for_asp_solver and i in self.map_id_list:
                clauses = gen.generate_clauses_for_facts_for_asp_solver(
                    i, fact, self.probabilistic_facts[fact])
            else:
                clauses = gen.generate_clauses_for_facts(fact)

            for c in clauses:
                self.lines_prob.append(c)
            i = i + 1

        i = 0
        for abd in self.abducibles:
            # kind of hack, refactor generate_clauses_for abducibles TODO
            clauses, _ = gen.generate_clauses_for_abducibles("abducible " + abd + ".", i)
            i = i + 1
            for c in clauses:
                self.lines_prob.append(c)

        return True


    def inference_to_mpe(self, from_string: str = "") -> 'tuple[str,int]':
        '''
        Adds 'map' before probabilistic facts.
        '''
        f = self.get_file_handler(from_string)
        parsed_program : str = ""
        n_vars = 0
        for line in f:
            if "::" in line and not line.strip().startswith('%'):
                line = f"map {line.strip()}"
                n_vars += 1
            parsed_program = parsed_program + line + "\n"
        return parsed_program, n_vars 


    def parse_input_learning(self, from_string: str = "") -> 'tuple[list[list[str]],list[list[str]],str,dict[str,float],int]':
        '''
        #example(pos,Id,'atom') where Id is the Id of the (partial) answer set and atom is the correspondent atom
        #test(IdList)
        #train(IdList)
        #program('program') where program is a set of clauses
        #learnable(atom) where atom is a probabilistic fact with init probability 0.5
        '''
        lines: list[str] = []

        if self.filename == "":
            lines = from_string.split('\n')
        else:
            fp = open(self.filename, "r")
            lines = fp.readlines()
            fp.close()

        i = 0
        program = ""
        # target = ""
        prob_facts_dict: dict[str, float] = dict()
        interpretations_dict: dict[int, list[str]] = dict()

        training_set: list[list[str]] = []
        test_set: list[list[str]] = []

        train_ids: list[int] = []
        test_ids: list[int] = []

        offset = 0

        while i < len(lines):
            lines[i] = lines[i].replace('\n', '')
            if lines[i].startswith("#program('"):
                i = i + 1
                while(not (lines[i].startswith("')."))):
                    program = program + lines[i]
                    # look for prob facts in the program that need to be considered
                    # in the dict but whose probabilities cannot be set
                    if '::' in lines[i]:
                        prob_fact = lines[i].split('::')[1].replace(
                            '\n', '').replace('.', '').replace(' ', '')
                        prob_facts_dict[prob_fact] = float(lines[i].split('::')[0])
                        offset = offset + 1
                    i = i + 1
            elif lines[i].startswith("#learnable("):
                ll = lines[i].split("#learnable(")
                name = ll[1].replace('\n', '')[:-2]
                prob_facts_dict[name] = 0.5
                i = i + 1
            elif lines[i].startswith("#positive("):
                ll = lines[i].split("#positive(")
                id_interpretation = int(ll[1].split(',')[0])
                atom = ll[1].replace('\n', '')[len(str(id_interpretation)) + 1: -2]
                if id_interpretation in interpretations_dict.keys():
                    interpretations_dict[id_interpretation].append(atom)
                else:
                    interpretations_dict[id_interpretation] = [atom]
                i = i + 1
            elif lines[i].startswith("#negative("):
                ll = lines[i].split("#negative(")
                id_interpretation = int(ll[1].split(',')[0])
                atom = ll[1].replace('\n', '')[len(str(id_interpretation)) + 1: -2]
                if id_interpretation in interpretations_dict.keys():
                    interpretations_dict[id_interpretation].append(f"not {atom}")
                else:
                    interpretations_dict[id_interpretation] = [f"not {atom}"]

                i = i + 1
            elif lines[i].startswith("#train("):
                ll = lines[i].split("#train(")
                train_ids = list(map(int, ll[1].replace('\n', '')[:-2].split(',')))
                i = i + 1
            elif lines[i].startswith("#test("):
                ll = lines[i].split("#test(")
                test_ids = list(map(int, ll[1].replace('\n', '')[:-2].split(',')))
                i = i + 1
            else:
                i = i + 1

        for id in train_ids:
            training_set.append(interpretations_dict[int(id)])

        for id in test_ids:
            test_set.append(interpretations_dict[int(id)])

        return training_set, test_set, program, prob_facts_dict, offset


    def check_reserved(self, line : str) -> None:
        '''
        Dummy check for reserved names (q, nq, e, ne)
        '''
        if line.startswith('q:-'):
            print_error_and_exit("q is a reserved fact")
        elif line.startswith('nq:-'):
            print_error_and_exit("nq is a reserved fact")
        elif line.startswith('e:-'):
            print_error_and_exit("e is a reserved fact")
        elif line.startswith('ne:-'):
            print_error_and_exit("ne is a reserved fact")


    def get_content_to_compute_minimal_set_facts(self) -> 'list[str]':
        '''
        Parameters:
            - None
        Returns:
            - str: program used to compute the minimal set of probabilistic
            facts to make the query true
        Behavior:
            generate the file to pass to ASP to compute the minimal set
            of probabilistic facts to make the query true
        '''
        if self.evidence == "":
            prog = self.lines_prob + [":- not " + self.query + "."]
        else:
            prog = self.lines_prob + [":- not " + self.evidence + "."]
        
        return prog


    def get_asp_program(self) -> 'list[str]':
        '''
        Parameters:
            - None
        Returns:
            - str: string representing the program that can be used to 
            compute lower and upper probability
        Behavior:
            returns a string that represent the ASP program where models 
            need to be computed
        '''
        if self.query:
            self.lines_prob.append(f"q:- {self.query}.")
            self.lines_prob.append("#show q/0.")
            self.lines_prob.append(f"nq:- not {self.query}.")
            self.lines_prob.append("#show nq/0.")

            if self.evidence:
                self.lines_prob.append(f"e:- {self.evidence}.")
                self.lines_prob.append("#show e/0.")
                self.lines_prob.append(f"ne:- not {self.evidence}.")
                self.lines_prob.append("#show ne/0.")

        return list(set(self.lines_prob))


    def get_asp_program_approx(self) -> 'list[str]':
        '''
        Parameters:
            - None
        Returns:
            - str: string representing the program that can be used to 
            compute lower and upper probability
        Behavior:
            returns a string that represent the ASP program where models 
            need to be computed
        '''
        if self.evidence == "":
            self.lines_prob.append(f"q:- {self.query}.")
            self.lines_prob.append("#show q/0.")
            self.lines_prob.append(f"nq:- not {self.query}.")
            self.lines_prob.append("#show nq/0.")
        else:
            self.lines_prob.append(f"qe:- {self.query}, {self.evidence}.")
            self.lines_prob.append("#show qe/0.")
            self.lines_prob.append(f"nqe:- not {self.query}, {self.evidence}.")
            self.lines_prob.append("#show nqe/0.")

        return list(set(self.lines_prob))


    def add_probabilistic_fact(self, term : str, prob : float) -> None:
        '''
        Adds the current probabilistic fact and its probability in the 
        list of probabilistic facts
        '''
        key = term.split('.')[0]
        if key in self.probabilistic_facts:
            error_prob_fact_twice(key, prob, self.probabilistic_facts[key])
        self.probabilistic_facts[key] = float(prob)


    def __repr__(self) -> str:
        '''
        String representation of the current class
        '''
        return "filename: " + self.filename + "\n" + \
        "query: " + str(self.query) + "\n" + \
        (("evidence: " + str(self.evidence) + "\n") if self.evidence else "") + \
        "probabilistic facts:\n" + str([str(x) + " " + str(y) for x, y in self.probabilistic_facts.items()]) + "\n" + \
        "n probabilistic facts:\n" + str(self.probabilistic_facts) + "\n" + \
        "original file:\n" + str(self.lines_original) + "\n" + \
        "probabilities file:\n" + str(self.lines_prob) + "\n" + \
        (("abducibles: " + str(self.abducibles)) if len(self.abducibles) > 0 else "")
