import unittest

import utils_for_tests as t_utils

import sys
sys.path.append("../pasta/lifted/")

# spec = importlib.util.spec_from_file_location("pasta", "../pasta/lifted.py")
# past = importlib.util.module_from_spec(spec) 
# spec.loader.exec_module(past)

# TODO, improve

import lifted

class TestClassLiftedInference(unittest.TestCase):
    
    # (c(X,Y) | a(X), b(X,Y))[lb,ub] multiple
    # def test_smoke_2_qr_exit(self):
    #     with pytest.raises(SystemExit):
    #         self.wrap_test_exact_inference("../examples/inference/smoke_2.lp", "qr", "", "smoke_2_qr", 0.055408970976253295, 0.13398746701846967)
    def test_cxy_ax_bxy_multiple_bi_1_lower(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_1_lower"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 3], lower=40)
        self.assertTrue(t_utils.almostEqual(lp, 0.0576, 5), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.16, 5), test_name + ": wrong upper probability")
    def test_cxy_ax_bxy_multiple_bi_1_upper(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_1_upper"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 3], lower=0, upper=70)
        self.assertTrue(t_utils.almostEqual(lp, 0, 5), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.10240, 5), test_name + ": wrong upper probability")


    def test_cxy_ax_bxy_multiple_bi_2_lower(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_2_lower"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 2, 2], lower=40)
        self.assertTrue(t_utils.almostEqual(lp, 0.071424, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.16, 5), test_name + ": wrong upper probability")
    def test_cxy_ax_bxy_multiple_bi_2_upper(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_2_upper"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 2, 2], lower=0, upper=70)
        self.assertTrue(t_utils.almostEqual(lp, 0, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.088576, 5), test_name + ": wrong upper probability")


    def test_cxy_ax_bxy_multiple_bi_3_lower(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_3_lower"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 2, 2, 1], lower=40)
        self.assertTrue(t_utils.almostEqual(lp, 0.059996160, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.16, 5), test_name + ": wrong upper probability")
    def test_cxy_ax_bxy_multiple_bi_3_upper(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_3_upper"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 2, 2, 1], lower=0, upper=70)
        self.assertTrue(t_utils.almostEqual(lp, 0, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.10000384, 8), test_name + ": wrong upper probability")


    def test_cxy_ax_bxy_multiple_bi_4_lower(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_4_lower"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 3, 2], lower=40)
        self.assertTrue(t_utils.almostEqual(lp, 0.0428544, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.16, 5), test_name + ": wrong upper probability")
    def test_cxy_ax_bxy_multiple_bi_4_upper(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_4_upper"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 3, 2], lower=0, upper=70)
        self.assertTrue(t_utils.almostEqual(lp, 0, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.1171456, 6), test_name + ": wrong upper probability")


    def test_cxy_ax_bxy_multiple_bi_5_lower(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_5_lower"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 3, 2, 1], lower=40)
        self.assertTrue(t_utils.almostEqual(lp, 0.035997696, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.16, 5), test_name + ": wrong upper probability")
    def test_cxy_ax_bxy_multiple_bi_5_upper(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_5_upper"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 3, 2, 1], lower=0, upper=70)
        self.assertTrue(t_utils.almostEqual(lp, 0, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.1240023, 6), test_name + ": wrong upper probability")


    def test_cxy_ax_bxy_multiple_bi_6_lower(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_6_lower"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1,1,1,1,1,3,2,2,1,1], lower=10)
        self.assertTrue(t_utils.almostEqual(lp, 0.02249712, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.16, 5), test_name + ": wrong upper probability")    
    def test_cxy_ax_bxy_multiple_bi_6_upper(self):
        test_name = "test_cxy_ax_bxy_multiple_bi_6_upper"
        lp, up, _, _ = lifted.cxy_ax_bxy_multiple_bi(0.4, [1,1,1,1,1,3,2,2,1,1], lower=0, upper=80)
        self.assertTrue(t_utils.almostEqual(lp, 0, 6), test_name + ": wrong lower probability")
        self.assertTrue(t_utils.almostEqual(up, 0.137502879, 5), test_name + ": wrong upper probability")
        
if __name__ == '__main__':
    unittest.main(buffer=True)
