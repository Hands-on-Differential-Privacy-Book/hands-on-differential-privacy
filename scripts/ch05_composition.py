from math import sqrt
import numpy as np
import opendp.prelude as dp
from utilities.analytic_gaussian import calibrateAnalyticGaussianMechanism
dp.enable_features("contrib")


def composition_approxDP_static_homo_basic(distance_0, k):
    """apply basic composition on `distance_0` in k-folds

    :param distance_0: per-query epsilon, delta
    :param k: how many folds, number of queries
    :returns global (epsilon, delta) of k-fold composition of a (epsilon_0, delta_0)-DP mechanism
    """
    epsilon_0, delta_0 = distance_0
    return epsilon_0 * k, delta_0 * k


def composition_approxDP_static_homo_advanced(distance_0, k, delta_p):
    """apply advanced composition on `distance_0` in k-folds

    "advanced" composition from Theorem 3.3 in https://guyrothblum.files.wordpress.com/2014/11/drv10.pdf
    Sometimes also referred to as "strong" composition.

    :param distance_0: per-query epsilon, delta
    :param k: how many folds, number of queries
    :param delta_p: how much additional delta to add, beyond basic composition of `delta_0`
    :returns global (epsilon, delta) of k-fold composition of a (epsilon_0, delta_0)-DP mechanism
    """
    eps_0, del_0 = distance_0
    eps_g = eps_0 * np.sqrt(2 * k * np.log(1 / delta_p)) + k * eps_0 * (
        np.exp(eps_0) - 1
    )
    del_g = del_0 * k + delta_p
    return eps_g, del_g


def composition_approxDP_static_homo_optimal_analytic(distance_0, k, delta_p):
    """apply composition on `distance_0` in k-folds

    "optimal" composition from KOV15
    "analytic" because this is the looser closed form expression in Theorem 3.4: https://arxiv.org/pdf/1311.0776.pdf#subsection.3.3

    :param distance_0: (epsilon, delta)
    :param delta_p: p as in prime. Slack term for delta. Allows for nontrivial epsilon composition
    """
    eps_0, del_0 = distance_0

    bound1 = k * eps_0
    t1 = (np.exp(eps_0) - 1) * eps_0 * k / (np.exp(eps_0) + 1)
    bound2 = t1 + eps_0 * np.sqrt(2 * k * np.log(np.e + eps_0 * np.sqrt(k) / delta_p))
    bound3 = t1 + eps_0 * np.sqrt(2 * k * np.log(1 / delta_p))

    # Corresponds to Theorem 3.4 in KOV15. Ignoring nan.
    eps_g = np.nanmin([bound1, bound2, bound3])
    del_g = 1 - (1 - delta_p) * (1 - del_0) ** k

    return eps_g, del_g


def composition_approxDP_static_homo_zCDP(distance_0, k):
    """apply composition on `distance_0` in k-folds

    :param distance_0: (epsilon, delta)
    :param delta_p: p as in prime. Slack term for delta. Allows for nontrivial epsilon composition
    """
    eps_0, del_0 = distance_0

    space = dp.atom_domain(T=float), dp.absolute_distance(T=float)
    scale_0 = dp.binary_search_param(
        lambda s: dp.c.make_fix_delta(
            dp.c.make_zCDP_to_approxDP(space >> dp.m.then_gaussian(s)), delta=del_0
        ),
        1.0,
        (eps_0, del_0),
    )

    del_g = del_0 * k
    eps_g = (
        dp.c.make_zCDP_to_approxDP(
            dp.c.make_basic_composition([space >> dp.m.then_gaussian(scale_0)] * k)
        )
        .map(1.0)
        .epsilon(delta=del_g)
    )

    return eps_g, del_g


def composition_approxDP_static_homo_plrv_analytic(distance_0, k):
    """apply composition on `distance_0` in k-folds

    :param distance_0: (epsilon, delta)
    :param delta_p: p as in prime. Slack term for delta. Allows for nontrivial epsilon composition
    """
    sens = 1.0
    eps_0, del_0 = distance_0

    scale_0 = calibrateAnalyticGaussianMechanism(eps_0, del_0, sens)
    eta_0 = sens**2 / (2 * scale_0**2)
    eta_g = eta_0 * k
    scale_g = sqrt(sens**2 / (2 * eta_g))

    del_g = del_0 * k
    eps_g = dp.binary_search(
        lambda e: calibrateAnalyticGaussianMechanism(e, del_g, 1.0) <= scale_g
    )

    return eps_g, del_g


epsilon_g = 1.0  # global epsilon (optimal performs best when sqrt(k) * epsilon << 1)
delta_g = 1e-6  # global delta

alpha = 0.5  # ratio of how much to allocate to δ' vs global δ


def compare(k, delta_g, alpha):
    # how much additional delta to allocate in advanced composition: δ'
    delta_p = delta_g * alpha

    compositors = {}

    delta_0_basic = delta_g / k
    epsilon_0_basic = epsilon_g / k
    compositors["basic"] = epsilon_0_basic, delta_0_basic

    delta_0_advanced = (delta_g - delta_p) / k
    epsilon_0_advanced = dp.binary_search(
        lambda e: composition_approxDP_static_homo_advanced(
            (e, delta_0_advanced), k, delta_p
        )[0]
        <= epsilon_g
    )
    compositors["advanced"] = epsilon_0_advanced, delta_0_advanced

    # # nearly indistinguishable from linear when delta small
    # delta_0_optimal = dp.binary_search(
    #     lambda d: composition_approxDP_static_homo_optimal_analytic(
    #         (1.0, d), k, delta_p
    #     )[1]
    #     <= delta_g
    # )
    # epsilon_0_optimal = dp.binary_search(
    #     lambda e: composition_approxDP_static_homo_optimal_analytic(
    #         (e, delta_0_optimal), k, delta_p
    #     )[0]
    #     <= epsilon_g
    # )
    # compositors["optimal"] = epsilon_0_optimal, delta_0_optimal

    delta_0_zCDP = delta_g / k
    epsilon_0_zCDP = dp.binary_search(
        lambda e: composition_approxDP_static_homo_zCDP((e, delta_0_zCDP), k)[0]
        <= epsilon_g
    )
    compositors["zCDP"] = epsilon_0_zCDP, delta_0_zCDP

    # delta_0_plrv = delta_g / k
    # epsilon_0_plrv = dp.binary_search(
    #     lambda e: composition_approxDP_static_homo_plrv_analytic((e, delta_0_plrv), k)[
    #         0
    #     ]
    #     <= epsilon_g,
    #     T=float,
    # )
    # compositors["PLRV"] = epsilon_0_plrv, delta_0_plrv

    return compositors

import matplotlib.pyplot as plt
import pandas as pd

all_dfs = []
for k in range(1, 101):
    print(k)
    comparison = compare(k, delta_g, alpha)
    legend = comparison.keys()
    epsilons, deltas = zip(*comparison.values())
    all_dfs.append(
        pd.DataFrame(
            {
                "k": [k] * len(comparison),
                "compositor": legend,
                "per-query effective $\epsilon$": epsilons,
                "per-query effective $\delta$": deltas,
            }
        )
    )
df = pd.concat(all_dfs)

import seaborn as sns


fig, ax = plt.subplots(1, 2)


df_epsilon = df.loc[~(df["compositor"] == "PLRV")]
# df_epsilon.loc[:, "compositor"] = (
#     df_epsilon["compositor"]
#     .replace("zCDP", "zCDP/PLRV")
# )
lp = sns.lineplot(
    data=df_epsilon, x="k", y="per-query effective $\epsilon$", hue="compositor", ax=ax[0]
)
lp.set(yscale="log")

df_delta = df.loc[df["compositor"].isin(["basic", "advanced"])]
df_delta.loc[:, "compositor"] = (
    df_delta["compositor"]
    .replace("basic", "basic/zCDP")
    # .replace("advanced", "advanced/optimal")
)
lp = sns.lineplot(
    data=df_delta, x="k", y="per-query effective $\delta$", hue="compositor", ax=ax[1]
)
lp.set(yscale="log")

fig.set_figwidth(10)
fig.set_figheight(4)
plt.savefig("../images/ch05_composition.png")
# plt.show()