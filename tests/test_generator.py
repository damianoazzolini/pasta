# from typing import Generator
# import unittest

# import importlib.util

# import sys
# from unittest import result
# sys.path.append("../pasta/")

# spec = importlib.util.spec_from_file_location(
#     "generator", "../pasta/generator.py")
# generat = importlib.util.module_from_spec(spec)
# spec.loader.exec_module(generat)


# class TestExtractVars(unittest.TestCase):
#     def test_correct(self):
#         self.assertCountEqual(["B","C"], generat.Generator.extract_vars("f(a,B,C)"), "Error")


# class TestExpandAbducible(unittest.TestCase):
#     def test_correct(self):
#         pass


# class TestExpandConditional(unittest.TestCase):
#     def test_correct_1(self):
#         res = generat.Generator.expand_conditional(
#             "(f(X)|h(X))[0.2,1].")

#         self.assertEqual(res[1],'',"Not empty lower")

#         del res[1]

#         res[0] = res[0].replace(' ','')
#         res[1] = res[1].replace(' ','')

#         e0 = "f(X);not_f(X):-h(X)."
#         e1 = ":-#count{X:f(X)}=H,#count{X:f(X),h(X)}=FH,100*FH<20*H."

#         self.assertCountEqual(res, [e0,e1], "Error")

#     # def test_correct_2(self):
#     #     res = generat.Generator.expand_conditional("(f2(X, Y )|h(Y )r(X, Y ))[0.9,1].")
#     #     expected = ["f2(X,Y) ; not_f2(X,Y) :- h(Y), r(X,Y).",":- #count{X:h(X),r(X,_)} = H, #count{Y,X:f2(X,Y),h(Y),r(X,Y)} = FH,10*FH < 9*H."]

#     #     self.assertCountEqual(res, expected, "Error")

#     def test_missing_pipe(self):
#         cond = "(f(X) h(X))[0.2,1]."
#         cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace('[','\[').replace(']','\]')

#         self.assertRaisesRegex(
#             SystemExit, "Syntax error in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

#     def test_missing_par(self):
#         cond = "(f(X) | h(X)[0.2,1]."
#         cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace(
#             '[', '\[').replace(']', '\]')


#         self.assertRaisesRegex(
#             SystemExit, "Syntax error in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

#     def test_missing_range(self):
#         cond = "(f(X) | h(X))."
#         cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace(
#             '[', '\[').replace(']', '\]')


#         self.assertRaisesRegex(
#             SystemExit, "Missing range in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

#     def test_unbalanced_range(self):
#         cond = "(f(X) | h(X))[0]."
#         cond_escaped = cond.replace('(', '\(').replace(')', '\)').replace(
#             '[', '\[').replace(']', '\]')

#         self.assertRaisesRegex(
#             SystemExit, "Unbalanced range in conditional: " + cond_escaped, generat.Generator.expand_conditional, cond)

#     def test_missing_final_dot(self):
#         cond = "(f(X) | h(X))[0]"

#         self.assertRaisesRegex(
#             SystemExit, "Missing final .", generat.Generator.expand_conditional, cond)


# if __name__ == '__main__':
#     unittest.main(buffer=True)
