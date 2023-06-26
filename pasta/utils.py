'''
Provides some utilities to print warnings and errors.
'''
import sys

YELLOW = '\33[93m'
RED = '\033[91m'
END = '\033[0m'


def print_error_and_exit(message : str):
    '''
    Prints the error message 'message' and exits.
    '''
    print(RED + "Error: " + message + END)
    sys.exit(-1)


def print_warning(message : str):
    '''
    Prints the warning message 'message'.
    '''
    print(YELLOW + "Warning: " + message + END)


def print_pathological_program():
    '''
    The program has 0 answer sets
    '''
    message = "This program is UNSAT (no answer sets)."
    print_warning(message)


def print_inconsistent_program(stop: bool = False):
    '''
    Prints that the program is inconsistent
    '''
    message = "This program is inconsistent. Use the --normalize flag."
    if stop:
        print_error_and_exit(message)
    print_warning(message)


def print_inconsistent_program_approx(stop: bool = False, world: str = ""):
    '''
    Prints that the program is inconsistent
    '''
    message = f"Found inconsistent world {world}.\nUse the --normalize flag (only for unconditional sampling)."
    if stop:
        raise Exception(message)
    print_warning(message)


def error_prob_fact_twice(
    key : str,
    new_prob : float,
    old_prob : float
    ) -> None:
    '''
    Prints a warning to indicate a probabilistic fact defined twice.
    '''
    print(f"Probabilistic fact {key} already defined")
    print(f"with probability {old_prob}.")
    print(f"Trying to replace it with probability {new_prob}.")
    sys.exit()


def is_number(n: 'int|float|str') -> bool:
    '''
    Returns true if the argument is a number, false otherwise.
    '''
    try:
        float(n)
    except:
        return False
    return True


def clean_term(term: str) -> 'tuple[str,bool]':
    '''
    Removes the suffixes used for abduction and decision thery.
    '''
    positive = True
    if term.startswith('not_'):
        term = term.split('not_')[1]
        positive = False

    if term.startswith('abd_'):
        term = term.split('abd_')[1]
    elif term.startswith('decision_'):
        term = term.split('decision_')[1]
    elif term.startswith('utility_'):
        term = term.split('utility_')[1]
    
    return term, positive


def sum_string_list(bl: 'list[str]') -> 'list[int]':
    '''
    Sums the elements in the same position of a list of 01-strings.
    Example: ["011","111"] -> [1,2,2]
    '''
    return list(map(lambda n : sum(int(x) for x in n), zip(*bl)))


def print_map_state(prob : float, atoms_list : 'list[list[str]]', n_map_vars : int) -> None:
    '''
    Prints the MAP/MPE state.
    '''
    map_op = len(atoms_list) > 0 and len(atoms_list[0]) == n_map_vars
    map_or_mpe = "MPE" if map_op else "MAP"
    print(f"{map_or_mpe}: {prob}\n{map_or_mpe} states: {len(atoms_list)}")
    for i, el in enumerate(atoms_list):
        print(f"State {i}: {el}")


def print_prob(lp : float, up : float, lpmln : bool = False) -> None:
    '''
    Prints the probability values.
    '''
    if not lpmln:
        if lp == up:
            print(f"Lower probability == upper probability for the query: {lp}")
        else:
            print(f"Lower probability for the query: {lp}")
            print(f"Upper probability for the query: {up}")
    else:
        print(f"Probability for the query: {lp}")


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

def print_result_abduction(
    lp : float,
    up : float,
    abd_exp : 'list[list[str]]',
    upper : bool = False) -> None:
    '''
    Prints the result for abduction.
    '''
    abd_exp_no_dup = remove_dominated_explanations(abd_exp)
    # abd_exp_no_dup = abd_exp
    if len(abd_exp_no_dup) > 0 and up != 0:
        if upper:
            print(f"Upper probability for the query: {up}")
        else:
            print_prob(lp, up)

    n_exp = sum(1 for ex in abd_exp_no_dup if len(ex) > 0)
    print(f"Abductive explanations: {n_exp}")

    index = 0
    for el in abd_exp_no_dup:
        if len(el) > 0:
            print(f"Explanation {index}")
            index = index + 1
            print(sorted(el))
