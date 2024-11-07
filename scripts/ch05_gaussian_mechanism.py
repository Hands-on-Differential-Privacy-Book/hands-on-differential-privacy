import opendp.prelude as dp
import numpy as np


def make_naive_gaussian_mechanism(scale, delta):
    dp.assert_features("contrib", "floating-point") # not floating-point-safe!
    assert scale >= 0 and 0 < delta < 1

    def privacy_map(b_in): # takes in l2-sensitivity
        assert b_in >= 0, "sensitivity must be non-negative"

        epsilon = np.sqrt(2 * np.log(1.25 / delta)) * b_in / scale
        assert epsilon <= 1., "this proof requires ε <= 1"

        return epsilon, delta

    return dp.m.make_user_measurement(
        input_domain=dp.vector_domain(dp.atom_domain(T=float)),
        input_metric=dp.l2_distance(T=float),
        output_measure=dp.fixed_smoothed_max_divergence(T=float),
        function=lambda data: np.random.gaussian(shift=data, scale=scale),
        privacy_map=privacy_map)


import opendp.prelude as dp
# define the space of floating-point vectors that differ in L2 distance
space = dp.vector_domain(dp.atom_domain(T=float)), dp.l2_distance(T=float)
# apply the mechanism over the space
gaussian_mechanism = space >> dp.m.then_gaussian(scale=1.)


# define the space of floating-point vectors that differ in L2 distance
space = dp.vector_domain(dp.atom_domain(T=int)), dp.l2_distance(T=float)
# apply the mechanism over the space
gaussian_mechanism = space >> dp.m.then_gaussian(scale=1.)
