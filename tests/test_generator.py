from typing import Generator
import unittest

import importlib.util

import sys
from unittest import result
sys.path.append("../src/pasta/")

spec = importlib.util.spec_from_file_location(
    "generator", "../src/pasta/generator.py")
generat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generat)

class TestExtractVars(unittest.TestCase):
    def test_correct(self):
        self.assertCountEqual(["B","C"], generat.Generator.extract_vars("f(a,B,C)"), "Error")

class TestExpandAbducible(unittest.TestCase):
    def test_correct(self):
        pass

# class TestToProlog(unittest.TestCase):
#     def test_fact(self):
#         line = "fly(1)."
#         res = generat.Generator.to_prolog(line)
#         self.assertEqual(line, res, "Error in test fact")

#     def test_disjunctive_clause(self):
#         line = "fly(X);nofly(X):- bird(X)."
#         expected = ["fly(X):-bird(X).","nofly(X):-bird(X)."]
#         res = generat.Generator.to_prolog(line)
#         self.assertEqual(expected, res, "Error in test disjunctive")


#     def test_disjunctive_clause_with_single_body_aggregate(self):
#         line = "fly(X);nofly(X):- bird(X), #count{X:fly_a(X),bird_a(X)} = FBA."
#         expected = ["newGoalInserted0:-fly_a(X),bird_a(X).",
#                     "fly(X):-bird(X),newGoalInserted0.",
#                     "nofly(X):-bird(X),newGoalInserted0."]
#         res = generat.Generator.to_prolog(line)
#         self.assertEqual(expected, res, "Error in test disjunctive single aggregate")

#     def test_disjunctive_clause_with_two_body_aggregate(self):
#         line = "fly(X);nofly(X):- bird(X), #count{X:fly_a(X),bird_a(X)} = FBA, #count{X:fly_a(X),bird_a(X)} = FBA."
#         expected = ['newGoalInserted0:-fly_a(X),bird_a(X).', 'newGoalInserted1:-fly_a(X),bird_a(X).',
#                     'fly(X):-bird(X),newGoalInserted0,newGoalInserted1.', 'nofly(X):-bird(X),newGoalInserted0,newGoalInserted1.']
#         res = generat.Generator.to_prolog(line)
#         self.assertEqual(expected, res, "Error in test disjunctive two aggregates")

#     def test_head_aggregate_no_body(self):
#         line = "0{a}1."
#         expected = ['a.']
#         res = generat.Generator.to_prolog(line)
#         self.assertEqual(
#             expected, res, "Error in test head aggregate no body")

#     def test_head_aggregate_with_body(self):
#         line = "0{a}1:- f(a)."
#         expected = ['a:-f(a).']
#         res = generat.Generator.to_prolog(line)
#         self.assertEqual(
#             expected, res, "Error in test head aggregate with body")

#     def test_integrity_constraint(self):
#         line = ":- f(a)."
#         expected = ['newGoalInserted0:-f(a).']
#         res = generat.Generator.to_prolog(line)
#         self.assertEqual(
#             expected, res, "Error in test integrity constraint")

class TestExpandConditional(unittest.TestCase):
    def test_correct_1(self):
        res = generat.Generator.expand_conditional(
            "(f(X)|h(X))[0.2,1].")

        self.assertEqual(res[1],'',"Not empty lower")

        del res[1]

        res[0] = res[0].replace(' ','')
        res[1] = res[1].replace(' ','')

        e0 = "f(X);not_f(X):-h(X)."
        e1 = ":-#count{X:f(X)}=H,#count{X:f(X),h(X)}=FH,100*FH<20*H."

        self.assertCountEqual(res, [e0,e1], "Error")

    # def test_correct_2(self):
    #     res = generat.Generator.expand_conditional("(f2(X, Y )|h(Y )r(X, Y ))[0.9,1].")
    #     expected = ["f2(X,Y) ; not_f2(X,Y) :- h(Y), r(X,Y).",":- #count{X:h(X),r(X,_)} = H, #count{Y,X:f2(X,Y),h(Y),r(X,Y)} = FH,10*FH < 9*H."]

    #     self.assertCountEqual(res, expected, "Error")

    def test_missing_pipe(self):
        cond = "(f(X) h(X))[0.2,1]."
        cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace('[','\[').replace(']','\]')

        self.assertRaisesRegex(
            SystemExit, "Syntax error in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

    def test_missing_par(self):
        cond = "(f(X) | h(X)[0.2,1]."
        cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace(
            '[', '\[').replace(']', '\]')


        self.assertRaisesRegex(
            SystemExit, "Syntax error in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

    def test_missing_range(self):
        cond = "(f(X) | h(X))."
        cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace(
            '[', '\[').replace(']', '\]')


        self.assertRaisesRegex(
            SystemExit, "Missing range in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

    def test_unbalanced_range(self):
        cond = "(f(X) | h(X))[0]."
        cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace(
            '[', '\[').replace(']', '\]')

        self.assertRaisesRegex(
            SystemExit, "Unbalanced range in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

    def test_missing_final_dot(self):
        cond = "(f(X) | h(X))[0]"

        self.assertRaisesRegex(
            SystemExit, "Missing final .", generat.Generator.expand_conditional, cond)




if __name__ == '__main__':
    unittest.main(buffer=True)
