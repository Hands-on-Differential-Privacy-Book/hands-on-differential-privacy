import matplotlib.pyplot as plt
import numpy as np
from utilities.densities import laplace_pdf

scale = 1.
b_in = 1.
alpha = 2.

lpdf = laplace_pdf(scale)
cont_support = np.linspace(-4, 4, num=1000)

# density
plt.plot(cont_support, lpdf(cont_support), label="$M(x) \sim \mathrm{Lap}(x, b)$")
# zero
plt.plot([0, 0], [0, lpdf(0)], linestyle='dashed', color="black", alpha=.5)

# probability mass exceeding -alpha
plt.plot([-alpha, -alpha], [0, lpdf(0)], linestyle='dashed', color="black")
plt.text(-alpha + .1, 0.01, f"$-\\alpha$")

# probability mass exceeding alpha
plt.plot([alpha, alpha], [0, lpdf(0)], linestyle='dashed', color="black")
plt.text(alpha - .3, 0.01, f"$\\alpha$")

# delta region
delta_region = lpdf(cont_support)
delta_region[np.bitwise_and(-alpha < cont_support, cont_support < alpha)] = 0
plt.fill_between(cont_support, np.zeros_like(delta_region), delta_region, alpha=0.5, label="$\leq \\beta$")

plt.xlabel("support ($y$)")
plt.ylabel("$\Pr[M(x) = y]$")
plt.legend()
plt.title("Accuracy of the Laplace Mechanism")
plt.savefig("../images/ch02_accuracy_laplace.png")
plt.show()

