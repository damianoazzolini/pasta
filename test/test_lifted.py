from .utils_for_tests import almost_equal

import pastasolver.lifted.lifted as lft


# (c(X,Y) | a(X), b(X,Y))[lb,ub] multiple
# def test_smoke_2_qr_exit(self):
#     with pytest.raises(SystemExit):
#         self.wrap_test_exact_inference("../examples/inference/smoke_2.lp", "qr", "", "smoke_2_qr", 0.055408970976253295, 0.13398746701846967)

def test_cxy_ax_bxy_multiple_bi_1_lower():
    test_name = "test_cxy_ax_bxy_multiple_bi_1_lower"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 3], lower=40)
    assert almost_equal(lp, 0.0576), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.16), test_name + ": wrong upper probability"
def test_cxy_ax_bxy_multiple_bi_1_upper():
    test_name = "test_cxy_ax_bxy_multiple_bi_1_upper"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 3], lower=0, upper=70)
    assert almost_equal(lp, 0), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.10240), test_name + ": wrong upper probability"

def test_cxy_ax_bxy_multiple_bi_2_lower():
    test_name = "test_cxy_ax_bxy_multiple_bi_2_lower"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 2, 2], lower=40)
    assert almost_equal(lp, 0.071424), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.16), test_name + ": wrong upper probability"
def test_cxy_ax_bxy_multiple_bi_2_upper():
    test_name = "test_cxy_ax_bxy_multiple_bi_2_upper"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 2, 2], lower=0, upper=70)
    assert almost_equal(lp, 0), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.088576), test_name + ": wrong upper probability"


def test_cxy_ax_bxy_multiple_bi_3_lower():
    test_name = "test_cxy_ax_bxy_multiple_bi_3_lower"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 2, 2, 1], lower=40)
    assert almost_equal(lp, 0.059996160), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.16), test_name + ": wrong upper probability"
def test_cxy_ax_bxy_multiple_bi_3_upper():
    test_name = "test_cxy_ax_bxy_multiple_bi_3_upper"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 2, 2, 1], lower=0, upper=70)
    assert almost_equal(lp, 0), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.10000384), test_name + ": wrong upper probability"


def test_cxy_ax_bxy_multiple_bi_4_lower():
    test_name = "test_cxy_ax_bxy_multiple_bi_4_lower"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 3, 2], lower=40)
    assert almost_equal(lp, 0.0428544), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.16), test_name + ": wrong upper probability"
def test_cxy_ax_bxy_multiple_bi_4_upper():
    test_name = "test_cxy_ax_bxy_multiple_bi_4_upper"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 3, 2], lower=0, upper=70)
    assert almost_equal(lp, 0), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.1171456), test_name + ": wrong upper probability"


def test_cxy_ax_bxy_multiple_bi_5_lower():
    test_name = "test_cxy_ax_bxy_multiple_bi_5_lower"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 3, 2, 1], lower=40)
    assert almost_equal(lp, 0.035997696), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.16), test_name + ": wrong upper probability"
def test_cxy_ax_bxy_multiple_bi_5_upper():
    test_name = "test_cxy_ax_bxy_multiple_bi_5_upper"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1, 1, 1, 3, 2, 1], lower=0, upper=70)
    assert almost_equal(lp, 0), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.1240023), test_name + ": wrong upper probability"


def test_cxy_ax_bxy_multiple_bi_6_lower():
    test_name = "test_cxy_ax_bxy_multiple_bi_6_lower"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1,1,1,1,1,3,2,2,1,1], lower=10)
    assert almost_equal(lp, 0.02249712), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.16), test_name + ": wrong upper probability"
def test_cxy_ax_bxy_multiple_bi_6_upper():
    test_name = "test_cxy_ax_bxy_multiple_bi_6_upper"
    lp, up, _, _ = lft.cxy_ax_bxy_multiple_bi(0.4, [1,1,1,1,1,3,2,2,1,1], lower=0, upper=80)
    assert almost_equal(lp, 0), test_name + ": wrong lower probability"
    assert almost_equal(up, 0.137502879), test_name + ": wrong upper probability"
