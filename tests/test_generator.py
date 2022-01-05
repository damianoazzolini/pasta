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

class TestExpandConditional(unittest.TestCase):
    def test_correct_1(self):
        res = generat.Generator.expand_conditional(
            "(f(X)|h(X))[0.2,1]")
        expected = ["f(X) ; not_f2(X) :- h(Y), r(X,Y).", ":- #count{X:h(X)}=H, #count{X:f1(X),h(X)}=FH, 10*FH < 2*H."]

        self.assertCountEqual(res, expected, "Error")

    def test_correct_2(self):
        res = generat.Generator.expand_conditional("(f2(X, Y )|h(Y )r(X, Y ))[0.9,1]")
        expected = ["f2(X,Y) ; not_f2(X,Y) :- h(Y), r(X,Y).",":- #count{X:h(X),r(X,_)} = H, #count{Y,X:f2(X,Y),h(Y),r(X,Y)} = FH,10*FH < 9*H."]

        self.assertCountEqual(res, expected, "Error")

    def test_missing_pipe(self):
        cond = "(f(X) h(X))[0.2,1]"

        self.assertRaisesRegex(
            SystemExit, "Syntax error in conditional: " + cond, generat.Generator.expand_conditional, cond)

    def test_missing_par(self):
        cond = "f(X) | h(X))[0.2,1]"

        self.assertRaisesRegex(
            SystemExit, "Syntax error in conditional: " + cond, generat.Generator.expand_conditional, cond)

    def test_missing_range(self):
        cond = "(f(X) | h(X))"

        self.assertRaisesRegex(
            SystemExit, "Missing range in conditional: " + cond, generat.Generator.expand_conditional, cond)

    def test_unbalanced_range(self):
        cond = "(f(X) | h(X))[0]"

        self.assertRaisesRegex(
            SystemExit, "Unbalanced range in conditional: " + cond, generat.Generator.expand_conditional, cond)




if __name__ == '__main__':
    unittest.main(buffer=True)
