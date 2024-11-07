import opendp.prelude as dp
import numpy as np


def renyi_divergence():
    return dp.user_divergence("RenyiDivergence()")


def make_RDP_to_fixed_approxDP(meas, delta):
    """Convert an RDP guarantee to an approx-DP guarantee"""
    assert meas.output_measure == renyi_divergence()

    def privacy_map(b_in):
        # (α, \bar{ε})-RDP implies (ε, δ)-DP
        rdp_curve = meas.map(b_in)

        # check some alphas and return the best ε
        alphas = np.arange(2, 200)
        epsilon = min(rdp_curve(a) + np.log(1 / delta) / (a - 1) for a in alphas)
        return epsilon, delta
    
    return dp.m.make_user_measurement(
        meas.input_domain, meas.input_metric, 
        dp.fixed_smoothed_max_divergence(T=float),
        function=meas.function, privacy_map=privacy_map)
