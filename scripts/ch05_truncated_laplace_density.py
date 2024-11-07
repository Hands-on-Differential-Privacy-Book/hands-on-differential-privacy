import numpy as np
import matplotlib.pyplot as plt
import opendp.prelude as dp
from utilities.densities import laplace_pdf, truncated_laplace_pdf
dp.enable_features("floating-point", "honest-but-curious", "contrib")

scale = 1.
radius = 2.
sensitivity = 0.5
support = np.linspace(-4, 4, num=300)



fig, ax = plt.subplots(1, 2)
densities_tlap = truncated_laplace_pdf(scale=scale, radius=radius)(support)
ax[0].plot(support, densities_tlap, label="$M(\cdot)$ is TLap(0)")
ax[0].plot(support, laplace_pdf(scale)(support), label="$M(\cdot)$ is Lap(0)")

ax[0].legend()
ax[0].set_xlabel("support (denoted $y$)")
ax[0].set_ylabel("density $\Pr[M(\cdot) = y]$")
ax[0].set_title("Laplace vs. TLaplace densities")


ax[1].plot(support, densities_tlap, label="$M(\cdot)$ is TLap(x)")
densities_tlap_p = truncated_laplace_pdf(scale=scale, radius=radius)(support + sensitivity)
ax[1].plot(support, densities_tlap_p, label="$M(\cdot)$ is TLap($x'$)")

tails = truncated_laplace_pdf(scale, radius, truncate=False)(support)
mask = np.bitwise_and(support >= radius - sensitivity, support <= radius)
ax[1].fill_between(support[mask], tails[mask], alpha=0.5, label="$\delta$")

ax[1].legend()
ax[1].set_xlabel("support (denoted $y$)")
ax[1].set_ylabel("density $\Pr[M(\cdot) = y]$")
ax[1].set_title("TLaplace densities on adjacent datasets")

fig.set_figwidth(10)
fig.set_figheight(4)
plt.savefig("../images/ch05_truncated_laplace.png")
plt.show()