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
    else:
        # this should not happen
        return -math.inf

def sample_gaussian(mean : float, stddev : float) -> float:
    '''
    Samples a value from the gaussian distribution with the specified
    mean and stddev.
    '''
    return float(scipy.stats.norm(mean, stddev).rvs(1))

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
        return -math.inf # only to make the linter happy, since the previous call calls sys exit

def sample_uniform(a : float, b : float) -> float:
    '''
    Samples a value from the uniform distribution with the specified
    loc and scale.
    '''
    return float(scipy.stats.uniform(loc=a, scale=b).rvs(1))

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
        return -math.inf # only to make the linter happy, since the previous call calls sys exit

def sample_exponential(lambda_rate : float) -> float:
    '''
    Samples a value from the exponential distribution with the specified
    rate.
    '''
    return float(scipy.stats.expon(scale = 1/ lambda_rate).rvs(1))


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
        return -math.inf # only to make the linter happy, since the previous call calls sys exit

def sample_gamma(alpha_shape : float, beta_rate : float) -> float:
    '''
    Samples a value from the gamma distribution with the specified
    alpha shape and beta rate.
    '''
    return float(scipy.stats.gamma(alpha_shape, scale = 1 / beta_rate).rvs(1))

def take_sample(distribution_parameters : 'tuple[str, float, float]') -> float:
    '''
    Take a samples from the specified distribution, dispatcher for the
    specific functions.
    '''
    distribution = distribution_parameters[0]
    if distribution == "gaussian":
        return sample_gaussian(distribution_parameters[1], distribution_parameters[2])
    elif distribution == "uniform":
        return sample_uniform(distribution_parameters[1], distribution_parameters[2])
    elif distribution == "exponential":
        return sample_exponential(distribution_parameters[1])
    elif distribution == "gamma":
        return sample_gamma(distribution_parameters[1], distribution_parameters[2])
    else:
        utils.print_error_and_exit(f"Distribution {distribution} not supported")
        return -math.inf # only to make the linter happy, since the previous call calls sys exit

def evaluate_sample(sample : float, fact : str) -> bool:
    # the variables have the form
    # var_type_before_after where
    # var is the name of the variable
    # type is in {above, below}
    # before is the integer part
    # after is the floating point part
    # example: a_below_0_5 == a < 0.5
    # replace 'minus' with - since 'minus' is inserted to process the program
    # in previous steps (parsing)
    f = fact.replace("minus","-").split('_') # ["a", "below", "0", "5"]
    if len(f) == 3:
        val = float(f[2])
    else:
        val = float(f[2] + '.' + f[3])

    comparison_type = f[1]
    if comparison_type == "above":
        return sample > val
    elif comparison_type == "below":
        return sample < val
    else:
        utils.print_error_and_exit("Sample to evaluate not conform")
        return False