import opendp.prelude as dp
from random import random, choice
import math

def make_randomized_response_multi(p: float, support: list):
    """
    :param p: probability of returning true answer
    :param support: all possible outcomes"""
    dp.assert_features("contrib", "floating-point")
    t = len(support)
    
    # CONDITIONS (see exercise 2)
    if t != len(set(support)):
        raise ValueError("elements in support must be distinct")
    if p < 1 / t or p > 1:
        raise ValueError(f"prob must be within [{1 / t}, 1.0]")
    
    def f_randomize_response(arg):
        lie = choice([x for x in support if arg != x])
        return arg if arg in support and random() < p else lie

    c = math.log(p / (1 - p) * (t - 1))

    return dp.m.make_user_measurement(
        input_domain=dp.atom_domain(T=type(support[0])),
        input_metric=dp.discrete_distance(),
        output_measure=dp.max_divergence(T=float),
        function=f_randomize_response,
        privacy_map=lambda b_in: min(max(b_in, 0), 1) * c,
        TO=type(support[0])
    )