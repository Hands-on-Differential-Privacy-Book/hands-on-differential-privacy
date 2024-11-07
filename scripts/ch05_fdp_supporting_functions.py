import numpy as np
import matplotlib.pyplot as plt
import opendp.prelude as dp

from utilities.analytic_gaussian import calibrateAnalyticGaussianMechanism

# now show a tradeoff curve of linear supporting functions

gauss_scale = 1.
sensitivity = 1.
alphas = np.linspace(0, 1, num=100)

def ag_εδ_curve(delta):
    """analytic gaussian mechanism ε(δ) curve"""
    return dp.binary_search(
        lambda e: calibrateAnalyticGaussianMechanism(e, delta, sensitivity) <= gauss_scale,
        T=float)

deltas = np.linspace(0.01, 1 / 3, 5)
approx_profile = [(ag_εδ_curve(d), d) for d in deltas]

for i, (epsilon, delta) in enumerate(approx_profile):
    linewidth = .5 if i == 2 else .5
    plt.plot(alphas, 1 - delta - np.exp(epsilon) * alphas, color="black", linewidth=linewidth)
    plt.plot(alphas, np.exp(-epsilon) * (1 - delta - alphas), color="black", linewidth=linewidth)

plt.gca().set_aspect("equal")
plt.xlabel("Type I error ($\\alpha$)")
plt.ylabel("Type II Error ($\\beta$)")
plt.xlim((0, 1))
plt.ylim((0, 1))

plt.savefig("../images/ch05_fdp_supporting_functions.png")
# plt.show()