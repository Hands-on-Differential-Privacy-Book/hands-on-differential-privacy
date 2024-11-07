import numpy as np
import opendp.prelude as dp
dp.enable_features("contrib", "honest-but-curious", "floating-point")


def make_interval_exponential_mechanism(bounds, scorer, entropy):
    L, U = bounds; assert L < U, "bounds must be increasing"
    def f_select_from_interval(x):
        # NEW: sort, clip and bookend x with bounds
        x = np.concatenate(([L], np.clip(np.sort(x), *bounds), [U]))

        scores = np.array(scorer(x)) # score all intervals in x
        scores -= scores.max() # for numerical stability; doesn't affect probs

        # NEW: area = width      * height; gives each interval's likelihood
        likelihoods = np.diff(x) * np.exp(scores * entropy)
        probabilities = likelihoods / likelihoods.sum() # normalize

        # use inverse transform sampling from the cdf to select an interval
        index = np.argmax(probabilities.cumsum() >= np.random.uniform())
        # NEW: sample uniformly from the selected interval
        return np.random.uniform(low=x[index], high=x[index + 1])
    
    mono = 1 if "monotonic" in str(scorer.output_metric) else 2
    return dp.m.make_user_measurement(
        input_domain=scorer.input_domain,
        input_metric=scorer.input_metric,
        output_measure=dp.max_divergence(T=float),
        function=f_select_from_interval,
        privacy_map=lambda b_in: scorer.map(b_in) / entropy * mono)


def make_score_quantile_interval(alpha, T=float):
    assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
    def f_score_quantile_interval(x):
        """Assuming `x` is sorted, scores each gap in `x`
        according to rank distance from the `alpha`-quantile."""
        ranks = np.arange(len(x) - 1)
        left, right = abs(ranks - alpha * len(x)), abs(ranks + 1 - alpha * len(x))
        return -np.minimum(left, right)
    
    return dp.t.make_user_transformation(
        input_domain=dp.vector_domain(dp.atom_domain(T=T)),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.linf_distance(T=float),
        function=f_score_quantile_interval,
        stability_map=lambda b_in: b_in * max(alpha, 1 - alpha))


def make_private_quantile_in_bounds(bounds, alpha, scale):
    scorer = make_score_quantile_interval(alpha)
    return make_interval_exponential_mechanism(bounds, scorer, scale)

meas = make_private_quantile_in_bounds(bounds=(0, 100.), alpha=.5, scale=1.)
print(meas(np.random.normal(22., scale=12, size=1000))) # ~> 21.37
print(meas.map(1)) # -> 1 = ε