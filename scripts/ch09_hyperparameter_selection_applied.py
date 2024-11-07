import numpy as np
import opendp.prelude as dp
from ch05_rdp_to_fixed_approx_dp import renyi_divergence
from ch06_measure_conversion import make_zCDP_to_RDP
from ch09_hyperparameter_selection import make_pspc_negative_binomial
from ch09_dpsgd_opacus import AdultDataSet

from torch.utils.data import Dataset
from opacus.accountants.analysis.rdp import compute_rdp


def make_count_correct():
    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(T=float, num_columns=2),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.atom_domain(T=float),
        output_metric=dp.absolute_distance(T=float),
        function=lambda x: sum(x[:, 0] == x[:, 1]),
        stability_map=lambda d_in: d_in,
    )


def torch_dataset_domain():
    return dp.user_domain("TorchDatasetDomain", lambda x: isinstance(x, Dataset))


def train_test_domain():
    def member(x):
        if not isinstance(x, tuple) or len(x) != 2:
            return False
        return all(isinstance(x_i, Dataset) for x_i in x)

    return dp.user_domain("TrainTestDomain", member)


def fit_adult_with_opacus(**kwargs):
    pass


def make_dpsgd_random_params(score_scale):
    m_acc = make_count_correct() >> dp.m.then_gaussian(score_scale)
    m_acc = make_zCDP_to_RDP(m_acc)

    def f_fit_score(data):
        hyperparams = dict(
            max_grad_norm=np.random.uniform(0.1, 10),
            hidden_size=np.random.randint(1, 500),
            learning_rate=np.random.exponential(0.05),
        )
        train, test = data
        model = fit_adult_with_opacus(train, **hyperparams)
        score = m_acc(np.stack([model(test.x), test.y], axis=1))
        return score, model, hyperparams
    # ...

    # ...
    def privacy_map(d_in):
        def rdp_curve(alpha):
            # these constants are from privacy_engine.accountant.history
            eps_train = compute_rdp(
                q=0.00033145508783559825,
                noise_multiplier=1.0,
                steps=30170,
                orders=alpha,
            )

            eps_test = m_acc.map(d_in)(alpha)
            return max(eps_train, eps_test)

        return rdp_curve

    return dp.m.make_user_measurement(
        input_domain=dp.np_array2_domain(T=float),
        input_metric=dp.symmetric_distance(),
        output_measure=renyi_divergence(),
        function=f_fit_score,
        privacy_map=privacy_map,
    )


m_dpsgd = make_dpsgd_random_params(score_scale=1.0)
m_dpsgd_pspc = make_pspc_negative_binomial(m_dpsgd, p=0.1, n=2)
data = AdultDataSet("adult.data"), AdultDataSet("adult.test")
score, model, hyperparams = m_dpsgd_pspc(data)
