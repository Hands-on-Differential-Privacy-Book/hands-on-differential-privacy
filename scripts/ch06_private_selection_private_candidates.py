import opendp.prelude as dp
import numpy as np


def make_pspc_geometric(meas, p):
    """implements pure-DP private selection from private candidates"""
    dp.assert_features("contrib", "floating-point")
    assert 0 < p < 1, "p is the probability of stopping"
    assert meas.output_measure == dp.max_divergence(T=float)

    def f_choose_best_run(data):
        # sample the geometric distribution- conditioned on not being zero!
        k = np.random.geometric(p) + 1 # geometric is memoryless

        # evaluate the measurement k times
        candidates = (meas(data) for _ in range(k))
        # select the candidate with the highest score
        return max(candidates, key=lambda c: c[0])

    return dp.m.make_user_measurement(
        *meas.input_space, dp.max_divergence(T=float),
        f_choose_best_run, privacy_map=lambda b_in: meas.map(b_in) * 3)
