
import numpy as np
import opendp.prelude as dp

from ch04_score_quantile_finite import make_score_quantile_finite


def make_finite_exponential_mechanism(temperature, monotonic=False):
    """Privately select the index of the best score from a vector of scores"""
    dp.assert_features("contrib", "floating-point")
    def f_select_index(scores):
        scores = np.array(scores)
        scores -= scores.max() # for numerical stability; doesn't affect probs

        likelihoods = np.exp(scores * temperature) # each candidate's likelihood
        probabilities = likelihoods / likelihoods.sum() # normalize
        # use inverse transform sampling from the cdf to select a candidate
        return np.argmax(probabilities.cumsum() >= np.random.uniform())
    
    return dp.m.make_user_measurement(
        input_domain=dp.vector_domain(dp.atom_domain(T=float)),
        input_metric=dp.linf_distance(T=float, monotonic=monotonic),
        output_measure=dp.max_divergence(T=float),
        function=f_select_index,
        privacy_map=lambda b_in: b_in / temperature * (1 if monotonic else 2))


def make_private_quantile_in_candidates(candidates, alpha, scale):
    return make_score_quantile_finite(candidates, alpha=alpha) >> \
        make_finite_exponential_mechanism(scale) >> \
        (lambda idx: candidates[idx]) # postprocess: retrieve the candidate


if __name__ == "__main__":
    dp.enable_features("contrib", "honest-but-curious", "floating-point")
    # meas = make_finite_exponential_mechanism(1.)
    # print(meas([1., 2., 3., 4.]))

    candidates = np.linspace(0, 100, num=21)
    meas = make_private_quantile_in_candidates(candidates, alpha=.5, scale=1.)
    print(meas(np.random.normal(22., scale=12, size=1000))) # ~> 21.37
    print(meas.map(1)) # -> 1 = ε