import math
import scipy.stats  # type: ignore

from . import utils

def evaluate_gaussian(mean : float, stddev : float, lb : float, ub : float) -> float:
    '''
    Evaluates P(lb < X < ub) where X follows a Gaussian distribution with
    mean mean and standard deviation stddev
    '''
    distr = scipy.stats.norm(mean, stddev)
    if lb == -math.inf and ub != math.inf:
        return distr.cdf(ub) # type: ignore
    elif lb != -math.inf and ub == math.inf:
        return 1 - distr.cdf(lb) # type: ignore
    elif lb != -math.inf and ub != math.inf:
        return distr.cdf(ub) - distr.cdf(lb)  # type: ignore


def evaluate_uniform(a : float, b : float, lb : float, ub : float) -> float:
    '''
    Evaluates P(lb < X < ub) where X follows a uniform distribution with
    lower bound a and upper bound b
    '''
    distr = scipy.stats.uniform(loc=a, scale=b)
    if lb == -math.inf and ub != math.inf:
        return distr.cdf(ub) # type: ignore
    elif lb != -math.inf and ub == math.inf:
        return 1 - distr.cdf(lb) # type: ignore
    elif lb != -math.inf and ub != math.inf:
        return distr.cdf(ub) - distr.cdf(lb) # type: ignore
    else:
        utils.print_error_and_exit("Both lb and ub are -inf")


def evaluate_exponential(lambda_rate : float, lb : float, ub : float) -> float:
    '''
    Evaluates P(lb < X < ub) where X follows an exponential distribution 
    with rate rate.
    '''
    distr = scipy.stats.expon(scale = 1/ lambda_rate)
    if lb == -math.inf and ub != math.inf:
        return distr.cdf(ub) # type: ignore
    elif lb != -math.inf and ub == math.inf:
        return 1 - distr.cdf(lb) # type: ignore
    elif lb != -math.inf and ub != math.inf:
        return distr.cdf(ub) - distr.cdf(lb)  # type: ignore
    else:
        utils.print_error_and_exit("Both lb and ub are -inf")


def evaluate_gamma(alpha_shape : float, beta_rate : float, lb : float, ub : float) -> float:
    '''
    Evaluates P(lb < X < ub) where X follows a gamma distribution 
    with alpha shape alpha_shape and beta rate beta_rate.
    '''
    distr = scipy.stats.gamma(alpha_shape, scale = 1 / beta_rate)
    if lb == -math.inf and ub != math.inf:
        return distr.cdf(ub) # type: ignore
    elif lb != -math.inf and ub == math.inf:
        return 1 - distr.cdf(lb) # type: ignore
    elif lb != -math.inf and ub != math.inf:
        return distr.cdf(ub) - distr.cdf(lb) # type: ignore
    else:
        utils.print_error_and_exit("Both lb and ub are -inf")
    