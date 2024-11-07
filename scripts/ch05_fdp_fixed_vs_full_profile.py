import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import opendp.prelude as dp

from utilities.analytic_gaussian import calibrateAnalyticGaussianMechanism

gauss_scale = 1.
sensitivity = 1.
alphas = np.linspace(0, 1, num=100)


def f_approxDP(profile):
    def curve(alpha):
        return max(max(
            0,
            1 - delta - np.exp(epsilon) * alpha,
            np.exp(-epsilon) * (1 - delta - alpha),
        ) for epsilon, delta in profile)
    return curve

def ag_εδ_curve(delta):
    """analytic gaussian mechanism ε(δ) curve"""
    if delta == 0:
        return float('inf')
    return dp.binary_search(
        lambda e: calibrateAnalyticGaussianMechanism(e, delta, sensitivity) <= gauss_scale,
        T=float)

delta_fixed = .1
fixed_profile = [(ag_εδ_curve(delta_fixed), delta_fixed)]
f_curve_fixed = f_approxDP(fixed_profile)
dfp_label = f"({fixed_profile[0][0]:.03}, {fixed_profile[0][1]:.03})-approxDP"

infinite_profile = np.linspace(0, 0.5, num=100)
f_curve_profile = f_approxDP([(ag_εδ_curve(d), d) for d in infinite_profile])


df = pd.DataFrame({
    "Type I Error ($\\alpha$)": alphas,
    # betas:
    "perfect privacy": 1 - alphas,
    dfp_label: [f_curve_fixed(a) for a in alphas],
    "gaussian profile": [f_curve_profile(a) for a in alphas],
})
lp = df.plot(x="Type I Error ($\\alpha$)", y=["perfect privacy", dfp_label, "gaussian profile"])
lp.set_aspect("equal")
lp.set_ylabel("Type II Error ($\\beta$)")
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.savefig("../images/ch05_fdp_fixed_vs_full_profile.png")
# plt.show()