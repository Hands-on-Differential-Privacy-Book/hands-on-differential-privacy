import numpy as np
import opendp.prelude as dp


def make_truncated_laplace(scale, radius):
    dp.assert_features("contrib", "floating-point") # not floating-point-safe!

    def f_add_tlap(arg):
        while True: # rejection sampling
            noise = np.random.laplace(scale=scale)
            if abs(noise) <= radius:
                return arg + noise

    def privacy_map(b_in):
        epsilon = b_in / scale
        delta = np.expm1(epsilon) / (np.expm1(radius / scale) * 2)
        return epsilon, delta
    
    return dp.m.make_user_measurement(
        input_domain=dp.atom_domain(T=float),
        input_metric=dp.absolute_distance(T=float),
        output_measure=dp.fixed_smoothed_max_divergence(T=float),
        function=f_add_tlap,
        privacy_map=privacy_map)


if __name__ == "__main__":
    dp.enable_features("floating-point", "honest-but-curious", "contrib")
    scale = 1.
    radius = 2.
    meas = make_truncated_laplace(scale, radius)
    meas(100.)
    print(meas.map(1.))
