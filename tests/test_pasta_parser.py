import unittest

import importlib.util

import sys
sys.path.append("../src/pasta/")

spec = importlib.util.spec_from_file_location("pasta_parser", "../src/pasta/pasta_parser.py")
pasta_parser = importlib.util.module_from_spec(spec)  # type: ignore
spec.loader.exec_module(pasta_parser)  # type: ignore

class TestEndlineContent(unittest.TestCase):
    def test_symbol_endline_or_space_n(self):
        self.assertEqual(pasta_parser.PastaParser.symbol_endline_or_space('\n'), True, "Endline content \\n not recognized")

    def test_symbol_endline_or_space_nr(self):
        self.assertEqual(pasta_parser.PastaParser.symbol_endline_or_space('\n\r'), False, "Endline content \\n\\r recognized")

    def test_symbol_endline_or_space_rn(self):
        self.assertEqual(pasta_parser.PastaParser.symbol_endline_or_space('\r\n'), True, "Endline content \\n\\r not recognized")

    def test_symbol_endline_or_space_space(self):
        self.assertEqual(pasta_parser.PastaParser.symbol_endline_or_space(' '), True, "Endline content space not recognized")

    def test_symbol_endline_or_space_fail(self):
        self.assertEqual(pasta_parser.PastaParser.symbol_endline_or_space('a'), False, "a recognized as endline content")


class TestEndlineComment(unittest.TestCase):
    def test_symbol_endline_or_space_n(self):
        self.assertEqual(pasta_parser.PastaParser.endline_symbol('\n'), True, "Endline comment \\n not recognized")

    def test_symbol_endline_or_space_nr(self):
        self.assertEqual(pasta_parser.PastaParser.endline_symbol('\n\r'), False, "Endline comment \\n\\r recognized")

    def test_symbol_endline_or_space_rn(self):
        self.assertEqual(pasta_parser.PastaParser.endline_symbol('\r\n'), True, "Endline comment \\n\\r not recognized")

    def test_symbol_endline_or_space_fail(self):
        self.assertEqual(pasta_parser.PastaParser.endline_symbol('a'), False, "a recognized as endline comment")


class TestIsNumber(unittest.TestCase):
    def test_is_int(self):
        self.assertEqual(pasta_parser.PastaParser.is_number(1), True,"1 not recognized as number")

    def test_is_int_string(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1"), True,"\"1\" (string) not recognized as number")

    def test_is_int_false_1(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1a"), False, "\"1a\" (string) recognized as number")

    def test_is_int_false_2(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("a1"), False, "\"a1\" (string) recognized as number")

    def test_is_float(self):
        self.assertEqual(pasta_parser.PastaParser.is_number(1.3), True, "1.3 not recognized as number")

    def test_is_float_string(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1.4"), True, "\"1.4\" (string) not recognized as number")

    def test_is_float_string_false_1(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1.4a"), False, "\"1.4a\" (string) recognized as number")

    def test_is_float_string_false_2(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("a1.4"), False, "\"a1.4\" (string) recognized as number")


class TestGetFunctor(unittest.TestCase):
    def test_atom(self):
        term = "a"
        self.assertEqual(pasta_parser.PastaParser.get_functor(term),term,term + "not recognized")

    def test_compound_arity_1(self):
        term = "a(f)"
        self.assertEqual(pasta_parser.PastaParser.get_functor(
            term), "a", term + "not recognized")

    def test_compound_arity_2(self):
        term = "a(f,b)"
        self.assertEqual(pasta_parser.PastaParser.get_functor(
            term), "a", term + "not recognized")
    
    def test_compound_range(self):
        term = "a(1..3)"
        self.assertEqual(pasta_parser.PastaParser.get_functor(
            term), "a", term + "not recognized")


class TestConsistencyProbFacts(unittest.TestCase):
    def test_correct_atom(self):
        pars = pasta_parser.PastaParser("test.pl",3)
        fact = "0.5::a."
        self.assertEqual(pars.check_consistent_prob_fact(fact), (0.5, "a"), fact + " not recognized as probabilistic fact")

    def test_correct_term(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.5::a(f)."
        self.assertEqual(pars.check_consistent_prob_fact(
            fact), (0.5, "a(f)"), fact + " not recognized as probabilistic fact")

    def test_wrong_prob(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.5q::a(f)."
        self.assertRaisesRegex(
            SystemExit, "Error: expected a float, found 0.5q", pars.check_consistent_prob_fact, fact)

    # def test_prob_above_1(self):
    #     pars = pasta_parser.PastaParser("test.pl", 3)
    #     fact = "1.3::a(f)."
    #     self.assertRaisesRegex(
    #         SystemExit, "Probabilities must be in the range [0,1], found 1.3", pars.check_consistent_prob_fact, fact)

    # def test_prob_0(self):
    #     pars = pasta_parser.PastaParser("test.pl", 3)
    #     fact = "0::a(f)."
    #     self.assertRaisesRegex(
    #         SystemExit, "Probabilities must be in the range ]0,1], found 0", pars.check_consistent_prob_fact, fact)

    def test_not_valid_term_first_number(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.3::2l."
        self.assertRaisesRegex(
            SystemExit, "Invalid probabilistic fact 2l", pars.check_consistent_prob_fact, fact)

    def test_not_valid_term_first_upper(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.3::Ul."
        self.assertRaisesRegex(
            SystemExit, "Invalid probabilistic fact Ul", pars.check_consistent_prob_fact, fact)

    def test_not_valid_term_symbol(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.3::()."
        self.assertRaisesRegex(
            SystemExit, "Invalid probabilistic fact ()", pars.check_consistent_prob_fact, fact)

    def test_not_valid_term_missing(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.3::."
        self.assertRaisesRegex(
            SystemExit, "Invalid probabilistic fact ", pars.check_consistent_prob_fact, fact)

    def test_missing_end_dot(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.5::a"
        self.assertRaisesRegex(SystemExit, "Missing final . in 0.5::a", pars.check_consistent_prob_fact, fact)

# class TestExtractAtomBetweenBrackets(unittest.TestCase):
#     def test_atom():
#         pass

#     def test_term():
#         pass

if __name__ == '__main__':
    unittest.main(buffer=True)
