import opendp.prelude as dp
import numpy as np

from ch05_rdp_to_fixed_approx_dp import renyi_divergence



def can_be_sampled(domain):
    # Not implemented
    _ = domain
    return True


def poisson_sample(data, q):
    # for each row in data, flip a coin where probability of 1 is q
    mask = np.random.binomial(n=1, p=q, size=len(data))
    # only keep rows where the coin flipped True
    return data[mask.astype(bool)]


def make_sampled_gaussian(trans, scale, q):
    dp.assert_features("contrib")
    assert can_be_sampled(trans.input_domain)
    assert trans.input_metric == dp.symmetric_distance()
    # privatize with the gaussian mechanism under RDP
    meas = trans >> dp.m.then_gaussian(scale)

    return dp.m.make_user_measurement(
        input_domain=trans.input_domain,
        input_metric=trans.input_metric,
        output_measure=renyi_divergence(),
        function=lambda data: meas(poisson_sample(data, q)),
        privacy_map=lambda b_in: sgm_privacy_map(b_in, meas, q)
    ) >> np.array


def sgm_privacy_map(b_in, meas, q):
    rho = meas.map(b_in)
    assert 0 < rho <= 1 and 0 < q < 0.5, "demo restriction"
    w = min(np.log(1 / q) / (4 * rho), 1 + q**(-1/4))
    assert w >= 3 + 2 * np.log(1 / rho) / np.log(1 / q), "demo restriction"

    # create a new RDP curve where loss is reduced by q^2
    def new_rdp_curve(alpha):
        assert 1 < alpha < w, "demo restriction"
        return 10 * q**2 * rho * alpha # FOCAL POINT
    return new_rdp_curve


# shadow the map with the tighter, more complicated bound (not in book, but in demo)
def sgm_privacy_map(b_in, meas, q):
    rho = meas.map(b_in)
    rdp_curve = lambda alpha: rho * alpha
    # create a new RDP curve where loss is reduced by q^2
    def new_rdp_curve(alpha):
        assert int(alpha) == alpha
        alpha = int(alpha)
        def binom(k):
            from math import comb
            return comb(alpha, k) * (1 - q)**(alpha - k) * q**k * np.exp((k - 1) * rdp_curve(k))
        
        t1 = (1 - q)**(alpha - 1) * (1 + (alpha - 1) * q)
        t2 = sum(binom(k) for k in range(2, alpha + 1))
        return np.log(t1 + t2) / (alpha - 1)
    return new_rdp_curve
