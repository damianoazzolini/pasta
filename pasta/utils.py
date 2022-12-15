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
        print_error_and_exit(message)
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
    try:
        float(n)
    except ValueError:
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


# progressbar from https://stackoverflow.com/questions/3160699/python-progress-bar
def progressbar(it: range, prefix: str = "", size: int = 60):
    count = len(it)

    def show(j: int):
        x = int(size*j/count)
        print(f"{prefix}[{u'█'*x}{('.'*(size-x))}] {j}/{count}",
              end='\r', file=sys.stdout, flush=True)

    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=sys.stdout)
