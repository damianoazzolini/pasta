import unittest

import importlib.util

spec = importlib.util.spec_from_file_location("utilities", "../src/pasta/utilities.py")
utilities = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utilities)

class TestConsistencyProbFacts(unittest.TestCase):
    def test_correct(self):
        fact = "0.5::a."
        self.assertEqual(utilities.check_consistent_prob_fact(fact), [0.5,"a"], fact + " not recognized as probabilistic fact")

    def test_wrong_prob(self):
        fact = "0.a::a."
        self.assertEqual(utilities.check_consistent_prob_fact(fact), [0.5, "a"], fact + " not recognized as probabilistic fact")


if __name__ == '__main__':
    unittest.main()
