import matplotlib.pyplot as plt
import numpy as np
from utilities.densities import gaussian_pdf

scale = 1.
b_in = 1.
epsilon = 2

eta = (b_in / scale)**2 / 2
shift = eta
gpdf = gaussian_pdf(2 * eta)
cont_support = np.linspace(-4, 4, num=1000)

# density
plt.plot(cont_support, gpdf(cont_support - shift), label="$\Pr[Z = z]$")
# zero
plt.plot([0, 0], [0, gpdf(0)], linestyle='dashed', color="black", alpha=.5)

# privacy loss exceeding -epsilon
plt.plot([-epsilon, -epsilon], [0, gpdf(0)], linestyle='dashed', color="black")
plt.text(-epsilon + .1, 0.01, f"$-\epsilon$")

# privacy loss exceeding epsilon
plt.plot([epsilon, epsilon], [0, gpdf(0)], linestyle='dashed', color="black")
plt.text(epsilon - .3, 0.01, f"$\epsilon$")

# delta region
delta_region = gpdf(cont_support - shift)
delta_region[np.bitwise_and(-epsilon < cont_support, cont_support < epsilon)] = 0
plt.fill_between(cont_support, np.zeros_like(delta_region), delta_region, alpha=0.5, label="$\geq \delta$")

plt.xlabel("support (y)")
plt.ylabel("$\mathcal{L}_{M(x) / M(x')}(y)$")
plt.legend()
plt.title("PLD of the Gaussian mechanism")
plt.savefig("../images/ch05_gaussian_plrv.png")
plt.show()

