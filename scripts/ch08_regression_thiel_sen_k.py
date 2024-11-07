from ch04_interval_exponential_mechanism import make_private_quantile_in_bounds
import opendp.prelude as dp
import numpy as np

alphas = np.array([[0.25], [0.75]])

def f_match(data):
    # keep an even number of rows
    data = np.array(data, copy=True)[:len(data) // 2 * 2]

    # evenly partition into random pairs
    np.random.shuffle(data)
    p1, p2 = np.array_split(data, 2)

    # compute a vector data set of slopes
    dx, dy = (p2 - p1).T
    slope = dy / dx

    # compute points on each line at 25th and 75th percentiles
    x_bar, y_bar = (p1 + p2).T / 2
    points = slope * (alphas - x_bar) + y_bar

    # only keep well-defined pairings
    return points.T[dx > 0]


def make_theil_sen_percentiles(k: int=1):
    dp.assert_features("contrib")
    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(T=float, num_columns=2),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.np_array2_domain(T=float, num_columns=2),
        output_metric=dp.symmetric_distance(),
        function=lambda x: np.vstack([f_match(x) for _ in range(k)]),
        stability_map=lambda b_in: k * b_in)


def make_select_column(j):
    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(T=float, num_columns=2),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.symmetric_distance(),
        function=lambda x: x[:, j],
        stability_map=lambda b_in: b_in)


def make_private_theil_sen(bounds, k, scale):
    # Released percentiles relate to regression coefficients by a system:
    #     p25 = .25 α + β
    #     p75 = .75 α + β
    # Equivalently:
    #     p = [[.25, 1.0], [.75, 1.0]] @ [[α], [β]]

    # Solve for the regression coefficients by inverting the system:
    def postprocess(p):
        return -2 * np.array([[1, -1], [-.75, .25]]) @ p
    
    m_median = make_private_quantile_in_bounds(bounds, 0.5, scale=scale)
    return make_theil_sen_percentiles(k) >> dp.c.make_basic_composition([
        make_select_column(0) >> m_median,
        make_select_column(1) >> m_median,
    ]) >> postprocess


dp.enable_features("honest-but-curious", "contrib")
meas = make_private_theil_sen((-10, 10), k=1, scale=1.)
x = np.random.normal(size=1_000)
y = 2 * x + 3 + np.random.normal(size=1_000)
data = np.stack([x, y], axis=1)

print(meas(data))
