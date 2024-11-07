import opendp.prelude as dp
import numpy as np

def make_score_quantile_finite(candidates, alpha):
    dp.assert_features("contrib")
    assert 0 <= alpha <= 1, "alpha must be in [0, 1]"
    assert len(set(candidates)) == len(candidates), "candidates must be unique"
    def f_score_candidates(x):
        """Assuming `x` is sorted, scores every element in `candidates`
        according to rank distance from the `alpha`-quantile."""
        num_leq = (np.array(x)[None] <= candidates[:, None]).sum(axis=1)
        return -abs(num_leq - alpha * len(x))
    
    return dp.t.make_user_transformation(
        input_domain=dp.vector_domain(dp.atom_domain(T=type(candidates[0]))),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.linf_distance(T=float),
        function=f_score_candidates,
        stability_map=lambda b_in: b_in * max(alpha, 1 - alpha))

if __name__ == "__main__":
    dp.enable_features("contrib", "honest-but-curious")
    trans = make_score_quantile_finite(np.linspace(0, 100, num=21), .5)

    print(trans(np.random.uniform(100, size=1000)))