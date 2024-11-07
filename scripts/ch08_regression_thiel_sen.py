from ch04_interval_exponential_mechanism import make_private_quantile_in_bounds
import opendp.prelude as dp
import numpy as np


def make_theil_sen_slopes():
    dp.assert_features("contrib")
    def f_compute_slopes(data):
         # keep an even number of rows
        data = np.array(data, copy=True)[:len(data) // 2 * 2]

        # evenly partition into random pairs
        np.random.shuffle(data)
        p1, p2 = np.array_split(data, 2)

        # compute a vector data set of slopes
        dx, dy = (p1 - p2).T
        return dy / dx

    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(T=float, num_columns=2),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.symmetric_distance(),
        function=f_compute_slopes,
        stability_map=lambda b_in: b_in)


def make_private_theil_sen(bounds, scale):
    return make_theil_sen_slopes() >> \
           make_private_quantile_in_bounds(bounds, alpha=0.5, scale=scale)


dp.enable_features("honest-but-curious", "contrib", "floating-point")
meas = make_private_theil_sen((-10, 10), 1.)
data = np.random.normal(size=(10_000, 2))
print(meas(data))
