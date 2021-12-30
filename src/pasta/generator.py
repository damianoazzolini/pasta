from typing import Union
'''
Class defining a generator of an ASP program
'''

class Generator:
    def __init__(self):
        pass

    # generates the dom fact
    # dom_f(1..3). from f(1..3).
    # dom_a from a
    @staticmethod
    def generate_dom_fact(functor: str, arguments: list) -> Union[str, str]:
        dom = "dom_" + functor
        args = ""
        if arguments[1] is False:  # single fact
            if arguments[0] is not None:
                for a in arguments[0]:
                    args = args + a + ","
                args = args[:-1]  # remove last ,
            else:
                args = None
        else:  # range
            # args = args + arguments[0][0] + ".." + arguments[0][1] + ")"
            args = "I"

        if arguments[1] is False:  # single fact
            dom_args = args
        else:  # range
            dom_args = arguments[0][0] + ".." + arguments[0][1]

        if args is not None:
            dom = dom + "(" + dom_args + ")."
        else:
            dom = dom + "."

        return dom, args

    @staticmethod
    def generate_generator(functor: str, args: str, arguments: list, prob: float, precision: int) -> Union[str, list]:
        if args is not None:
            vt = "v_" + functor + "_(" + args + ")"
        else:
            vt = "v_" + functor

        generator = ""
        generator = generator + "0{"
        if arguments[1] is False:
            number = 1
        else:
            number = int(arguments[0][1]) - int(arguments[0][0]) + 1

        if args is not None:
            generator = generator + vt + ":dom_" + functor + \
                "(" + args + ")}" + str(number) + "."
        else:
            generator = generator + vt + ":dom_" + \
                functor + "}" + str(number) + "."

        # generate the two clauses
        clauses = []
        prob_true = int(prob * (10**precision))
        prob_false = int((1 - prob) * (10**precision))

        if number != 1:  # clauses for range
            start = int(arguments[0][0])
            end = int(arguments[0][1])

            for i in range(start, end + 1):
                vt = "v_" + functor + "_(" + str(i) + ")"
                clause_true = functor + \
                    "(" + str(i) + "," + str(prob_true) + ")" + ":-" + vt + "."
                clause_false = "not_" + functor + \
                    "(" + str(i) + "," + str(prob_false) + ")" + ":- not " + vt + "."
                clauses.append(clause_true)
                clauses.append(clause_false)

            # add auxiliary clauses that wraps probabilities
            auxiliary_clause_true = functor + "(I) :- " + functor + "(I,_)."
            clauses.append(auxiliary_clause_true)

            auxiliary_clause_false = "not_" + functor + \
                "(I) :- not_" + functor + "(I,_)."
            clauses.append(auxiliary_clause_false)

            # add show declaration
            show_declaration = "#show " + functor + "/2."
            clauses.append(show_declaration)

            show_declaration = "#show not_" + functor + "/2."
            clauses.append(show_declaration)

        else:
            if args is not None:
                clause_true = functor + \
                    "(" + args + "," + str(prob_true) + ")" + ":-" + vt + "."
                clause_false = "not_" + functor + \
                    "(" + args + "," + str(prob_false) + ")" + ":- not " + vt + "."
                auxiliary_clause_true = functor + \
                    "(" + args + ") :- " + functor + "(" + args + ",_)."
                auxiliary_clause_false = "not_" + functor + \
                    "(" + args + ") :- not_" + functor + "(" + args + ",_)."
                show_declaration_t = "#show " + functor + \
                    "/" + str(args.count(',') + 2) + "."
                show_declaration_f = "#show not_" + functor + \
                    "/" + str(args.count(',') + 2) + "."
            else:
                clause_true = functor + \
                    "(" + str(prob_true) + ")" + ":-" + vt + "."
                clause_false = "not_" + functor + \
                    "(" + str(prob_false) + ")" + ":- not " + vt + "."
                auxiliary_clause_true = functor + ":- " + functor + "(_)."
                auxiliary_clause_false = "not_" + \
                    functor + ":- not_" + functor + "(_)."
                show_declaration_t = "#show " + functor + "/1."
                show_declaration_f = "#show not_" + functor + "/1."

            clauses.append(clause_true)
            clauses.append(clause_false)

            clauses.append(auxiliary_clause_true)
            clauses.append(auxiliary_clause_false)

            clauses.append(show_declaration_t)
            clauses.append(show_declaration_f)

        return generator, clauses