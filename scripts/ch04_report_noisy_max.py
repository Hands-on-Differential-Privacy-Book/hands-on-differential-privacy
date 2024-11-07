import numpy as np
import opendp.prelude as dp


def make_report_noisy_max_gumbel(scale, monotonic=False, T=float):
    dp.assert_features("contrib")
    assert scale >= 0, "scale must not be negative"
    return dp.m.make_user_measurement(
        input_domain=dp.vector_domain(dp.atom_domain(T=T)),
        input_metric=dp.linf_distance(T=T),
        output_measure=dp.max_divergence(T=float),
        # the value with the largest noisy score is the selected bin index
        function=lambda scores: np.argmax(np.random.gumbel(scores, scale=scale)),
        privacy_map=lambda b_in: b_in / scale * (1 if monotonic else 2))

def make_report_noisy_max_exponential(scale, monotonic=False, T=float):
    dp.assert_features("contrib")
    assert scale >= 0, "scale must not be negative"
    return dp.m.make_user_measurement(
        input_domain=dp.vector_domain(dp.atom_domain(T=T)),
        input_metric=dp.linf_distance(T=T),
        output_measure=dp.max_divergence(T=float),
        # the value with the largest noisy score is the selected bin index
        function=lambda scores: np.argmax(
            scores + np.random.exponential(scale=scale, size=len(scores))),
        privacy_map=lambda b_in: b_in / scale * (1 if monotonic else 2))


dp.enable_features("contrib", "honest-but-curious", "floating-point")
meas = make_report_noisy_max_gumbel(1.)
print(meas(np.random.normal(22., scale=12, size=1000))) # ~> 21.37
print(meas.map(1)) # -> 1 = ε