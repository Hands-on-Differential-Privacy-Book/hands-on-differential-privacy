import opendp.prelude as dp

from ch04_interval_exponential_mechanism import make_private_quantile_in_bounds


# See Algorithm 1: https://arxiv.org/pdf/2110.05429.pdf
# Takes advantage of information gained in prior quantile estimates.
def make_private_multi_quantile_in_bounds(bounds, alphas, scale):
    dp.assert_features("contrib")
    def f_recursive_quantiles(x, alphas, bounds, epsilon):
        # base cases
        if len(alphas) == 0:
            return [] # for when the tree is not full

        if len(alphas) == 1:
            return [make_private_quantile_in_bounds(bounds, alphas[0], scale)(x)]

        # always estimate the middle quantile
        mid = (len(alphas) + 1) // 2
        p = alphas[mid]
        v = make_private_quantile_in_bounds(bounds, p, scale)(x)

        # split x and alphas apart into sub-problems (while rescaling alphas)
        x_l, x_u = x[x < v], x[x > v]
        alphas_l, alphas_u = alphas[:mid] / p, (alphas[mid + 1 :] - p) / (1 - p)

        # recurse down left and right partitions
        return [
            *f_recursive_quantiles(x_l, alphas_l, (bounds[0], v), epsilon),
            v,
            *f_recursive_quantiles(x_u, alphas_u, (v, bounds[1]), epsilon),
        ]
    # ...
    
    # ...
    def privacy_map(b_in):
        # the per-partition loss is maximized when alpha is 1
        per_part_loss = make_private_quantile_in_bounds(bounds, 1., 
                                                        scale).map(b_in)
        # by parallel composition up to b_in partitions can be 
        # influenced per-layer
        per_layer_loss = b_in * per_part_loss
        # the recursion depth/number of layers is ceil(log_2(|alphas|))
        num_layers = len(alphas).bit_length()
        # by sequential composition over the releases of each layer
        return float(per_layer_loss * num_layers)

    return dp.m.make_user_measurement(
        input_domain=dp.vector_domain(dp.atom_domain(T=float)),
        input_metric=dp.symmetric_distance(),
        output_measure=dp.max_divergence(T=float),
        function=lambda x: f_recursive_quantiles(np.array(x), alphas, bounds, 
        scale),
        privacy_map=privacy_map)


import numpy as np
alphas = np.array([0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9])
bounds = (0., 100.)
scale = 100.
m_mq = make_private_multi_quantile_in_bounds(bounds, alphas, scale)

data = np.random.uniform(0, 100, size=10_000)
print(m_mq(data)) # ~> [9.735, 25.518, 40.292, 50.388, 59.679, 74.757, 89.544]
print(m_mq.map(1)) # -> 0.06 = ε
