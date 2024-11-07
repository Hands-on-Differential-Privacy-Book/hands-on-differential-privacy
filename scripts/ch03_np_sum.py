from ch03_np_clamp import make_np_clamp
import numpy as np
import opendp.prelude as dp


def make_np_sum(norm, p, origin=None):
    dp.assert_features("contrib", "floating-point")
    assert norm >= 0, "norm must not be negative"

    # assume the origin is at zero if not specified
    origin = 0.0 if origin is None else origin

    #      C = ||O||_p                                      + R
    constant = np.linalg.norm(np.atleast_1d(origin), ord=p) + norm
    
    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(norm=norm, p=p, origin=origin),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric={1: dp.l1_distance, 2: dp.l2_distance}[p](T=float),
        function=lambda data: data.sum(axis=0), 
        stability_map=lambda b_in: b_in * constant)


if __name__ == "__main__":
    dp.enable_features("contrib", "honest-but-curious", "floating-point")

    meas = make_np_clamp(norm=4., p=1) >> \
           make_np_sum(norm=4., p=1) >> \
           dp.m.then_laplace(scale=4.)

    meas(arg=np.ones((1_000, 4))) # ~> [253.22, 245.95, 247.87, 246.22]
    meas.map(1)              # -> 1 = ε