import numpy as np
import opendp.prelude as dp

from ch05_truncated_laplace import make_truncated_laplace
from ch04_rr_multi import make_randomized_response_multi


def get_elements(mechanisms):
    # ensure that all mechanisms have homogeneous...
    input_domain, = {m.input_domain for m in mechanisms} # ...input domain,
    input_metric, = {m.input_metric for m in mechanisms} # ...input metric,
    output_measure, = {m.output_measure for m in mechanisms} # ...and measure

    return input_domain, input_metric, output_measure


def make_advanced_composition(mechanisms, delta_p):
    """construct an advanced composition mechanism"""
    dp.assert_features("contrib", "floating-point")

    input_domain, input_metric, output_measure = get_elements(mechanisms)

    # ensure that the privacy measure is approx-DP
    assert output_measure == dp.fixed_smoothed_max_divergence(T=float)

    def privacy_map(b_in):
        epsilons, deltas = zip(*(M.map(b_in) for M in mechanisms))
        # respect the assumption that epsilons and deltas are homogeneous
        #     (this is very loose when ε_i, δ_i are heterogeneous)
        eps_0, del_0, k = max(epsilons), max(deltas), len(mechanisms)

        # t =  sqrt(2 * k *     ln(1 / δ'     )) + k * (   exp( ε_0 ) - 1)
        t = np.sqrt(2 * k * np.log(1 / delta_p)) + k * (np.exp(eps_0) - 1)

        #     (ε_0   * t, δ_0   * k + δ')
        return eps_0 * t, del_0 * k + delta_p
    
    return dp.m.make_user_measurement(
        input_domain, input_metric, output_measure,
        function=lambda arg: [M(arg) for M in mechanisms], 
        privacy_map=privacy_map)


if __name__ == "__main__":

    dp.enable_features("contrib", "honest-but-curious", "floating-point")

    get_elements([ # create a sequence of k=2 mechanisms with different domains
        make_randomized_response_multi(p=.7, support=["A", "B", "C"]),
        make_truncated_laplace(scale=1., radius=10.)
    ])


    domain = dp.vector_domain(dp.atom_domain(T=float), size=10_000)
    t_clamp = (domain, dp.symmetric_distance()) >> dp.t.then_clamp((0., 10.))

    M_i = [ # create a sequence of k=2 mechanisms
        t_clamp >> dp.t.then_mean() >> make_truncated_laplace(scale=.1,
                   radius=1.),
        t_clamp >> dp.t.then_variance() >> make_truncated_laplace(scale=1.,
                   radius=10.)
    ]

    m_ac = make_advanced_composition(M_i, delta_p = 1e-7)
    print(m_ac.map(2)) # -> (0.08, 5.5629e-07) = (ε, δ)

    m_bc = dp.c.make_basic_composition(M_i)
    print(m_bc.map(2)) # -> (0.02, 4.5629e-07) = (ε, δ)
