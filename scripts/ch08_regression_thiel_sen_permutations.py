from ch04_interval_exponential_mechanism import make_private_quantile_in_bounds
import opendp.prelude as dp
from itertools import permutations

dp.enable_features("honest-but-curious", "contrib")


def make_theil_sen_slopes(size):
    def slope(p0, p1):
        return (p1[1] - p0[1]) / (p1[0] - p0[0])

    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(T=float, num_columns=2, size=size),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.symmetric_distance(),
        function=lambda x: [slope(p0, p1) for p0, p1 in permutations(x, 2)],
        # each point influences n - 1 slopes
        stability_map=lambda b_in: (size - 1) * b_in,
    )


def make_theil_sen(size, bounds, scale):
    return make_theil_sen_slopes(size) >> make_private_quantile_in_bounds(
        bounds, 0.5, scale=scale
    )


def dp_theil_sen(size, bounds, epsilon):
    return dp.binary_search_chain(
        lambda s: make_theil_sen(size, bounds, s), d_in=1, d_out=epsilon
    )

import numpy as np

# Create an initial dataset of 1000 data points
x = np.arange(0, 1000, 1)
# Use x as an input to a linear equation, with some unknown error
y = 2 * x + 3 + np.random.normal(0., scale=1., size=len(x))
data = np.stack([x, y], axis=1)

m_theil_sen = dp_theil_sen(size=1000, bounds=(0, 10), epsilon=1.)
print(m_theil_sen(data))
