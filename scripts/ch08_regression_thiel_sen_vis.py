from ch04_interval_exponential_mechanism import make_private_quantile_in_bounds
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

np.random.seed(1)

n_rows = 20

# force n_rows to be even
n_rows -= n_rows % 2

# generate data
x = np.random.normal(size=n_rows)
y = x * 2 + 3 + np.random.normal(loc=0., scale=0.4, size=n_rows)
data = np.stack([x, y]).T

# randomly construct pairings
np.random.shuffle(data)
p1, p2 = np.array_split(data, 2)

# begin plotting facets
fig, ax = plt.subplots(1, 3) # , gridspec_kw={'width_ratios': [3, 1, 3]}

# ax[0]
for v1, v2 in zip(p1, p2):
    ax[0].plot([v1[0], v2[0]], [v1[1], v2[1]])
ax[0].set_title("random pairs")

# ax[1]
dx, dy = (p1 - p2).T
slopes = dy / dx

density = gaussian_kde(slopes)
support = np.linspace(0,5,200)
density.covariance_factor = lambda : 0.2
density._compute_covariance()
ax[1].eventplot(slopes, lineoffsets=-.06, linelengths=.1, orientation='horizontal')
ax[1].fill_between(support, density(support))
ax[1].get_yaxis().set_visible(False)
ax[1].set_xlim(0, 4)
ax[1].set_ylim(-.1, 1)

meas = make_private_quantile_in_bounds(bounds=(0, 4), alpha=.5, scale=1.)
slope = meas(slopes)

ax[1].vlines(slope, -.1, 1, color="red")
ax[1].set_title("dataset of slopes")

# ax[2]
ax[2].scatter(data[:, 0], data[:, 1])
ax[2].plot(x, x * slope + np.mean(y), color="red", label="fit")
ax[2].set_title("fitted model")

fig.set_figwidth(8)
fig.set_figheight(3)
plt.tight_layout()
plt.savefig("../images/ch08_regression_theil_sen.png")
plt.show()