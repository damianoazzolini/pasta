import unittest

import importlib.util

import sys
sys.path.append("../src/pasta/")

spec = importlib.util.spec_from_file_location("pasta_parser", "../src/pasta/pasta_parser.py")
pasta_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pasta_parser)

class TestEndlineContent(unittest.TestCase):
    def test_endline_content_n(self):
        self.assertEqual(pasta_parser.PastaParser.endline_content('\n'), True,
                         "Endline content \\n not recognized")

    def test_endline_content_nr(self):
        self.assertEqual(pasta_parser.PastaParser.endline_content('\n\r'),
                         False, "Endline content \\n\\r recognized")

    def test_endline_content_rn(self):
        self.assertEqual(pasta_parser.PastaParser.endline_content('\r\n'),
                         True, "Endline content \\n\\r not recognized")

    def test_endline_content_space(self):
        self.assertEqual(pasta_parser.PastaParser.endline_content(' '), True,
                         "Endline content space not recognized")

    def test_endline_content_fail(self):
        self.assertEqual(pasta_parser.PastaParser.endline_content(
            'a'), False, "a recognized as endline content")


class TestEndlineComment(unittest.TestCase):
    def test_endline_content_n(self):
        self.assertEqual(pasta_parser.PastaParser.endline_comment('\n'), True,
                         "Endline comment \\n not recognized")

    def test_endline_content_nr(self):
        self.assertEqual(pasta_parser.PastaParser.endline_comment('\n\r'),
                         False, "Endline comment \\n\\r recognized")

    def test_endline_content_rn(self):
        self.assertEqual(pasta_parser.PastaParser.endline_comment('\r\n'),
                         True, "Endline comment \\n\\r not recognized")

    def test_endline_content_fail(self):
        self.assertEqual(pasta_parser.PastaParser.endline_comment(
            'a'), False, "a recognized as endline comment")


class TestIsNumber(unittest.TestCase):
    def test_is_int(self):
        self.assertEqual(pasta_parser.PastaParser.is_number(1), True,
                         "1 not recognized as number")

    def test_is_int_string(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1"), True,
                         "\"1\" (string) not recognized as number")

    def test_is_int_false_1(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1a"), False,
                         "\"1a\" (string) recognized as number")

    def test_is_int_false_2(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("a1"), False,
                         "\"a1\" (string) recognized as number")

    def test_is_float(self):
        self.assertEqual(pasta_parser.PastaParser.is_number(1.3), True,
                         "1.3 not recognized as number")

    def test_is_float_string(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1.4"), True,
                         "\"1.4\" (string) not recognized as number")

    def test_is_float_string_false_1(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("1.4a"), False,
                         "\"1.4a\" (string) recognized as number")

    def test_is_float_string_false_2(self):
        self.assertEqual(pasta_parser.PastaParser.is_number("a1.4"), False,
                         "\"a1.4\" (string) recognized as number")


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

    def test_correct(self):
        pars = pasta_parser.PastaParser("test.pl",3)
        fact = "0.5::a."
        self.assertEqual(pars.check_consistent_prob_fact(fact), (0.5, "a"), fact + " not recognized as probabilistic fact")

    def test_missing_dot(self):
        pars = pasta_parser.PastaParser("test.pl", 3)
        fact = "0.5::a"
        self.assertRaisesRegex(SystemExit, "Missing ending . in 0.5::a", pars.check_consistent_prob_fact, fact)

    # def test_wrong_prob(self):
    #     fact = "0.a::a."
    #     self.assertEqual(utilities.check_consistent_prob_fact(fact), [
    #                      0.5, "a"], fact + " not recognized as probabilistic fact")



if __name__ == '__main__':
    unittest.main(buffer=True)
