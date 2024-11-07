import numpy as np
import opendp.prelude as dp
from ch05_rdp_to_fixed_approx_dp import renyi_divergence


def make_pspc_negative_binomial(meas, p, n=1):
    """implements RDP private selection from private candidates"""
    dp.assert_features("contrib", "floating-point")
    assert 0 < p < 1 and n > 0, "p is a probability and n must be positive"
    assert meas.output_measure == renyi_divergence()

    def f_choose_best_run(data):
        # sample the negative binomial distribution- 
        # conditioned on not being zero!
        k = 0
        while k == 0:
            k = np.random.negative_binomial(n, p)
        
        # evaluate the measurement k times
        candidates = (meas(data) for _ in range(k))
        # select the candidate with the highest score
        return max(candidates, key=lambda c: c[0])

    return dp.m.make_user_measurement(
        *meas.input_space, 
        renyi_divergence(T=float), f_choose_best_run, 
        privacy_map=lambda b_in: pspc_nb_privacy_map(b_in, meas, n, p))


def pspc_nb_privacy_map(b_in, meas, n, p):
    # construct a new curve that is less private
    rdp_curve = meas.map(b_in)
    def new_rdp_curve(alpha):
        assert alpha > 1, "RDP order (alpha) must be greater than one"
        eps = rdp_curve(alpha)
        t1 = (1 + n) * (1 - 1 / alpha) * eps
        t2 = (1 + p) * np.log(1 / p) / alpha
        t3 = n * (1 - p) / (p * (1 - p**n)) / (alpha - 1)
        return eps + t1 + t2 + t3
    return new_rdp_curve

