import numpy as np
import matplotlib.pyplot as plt
from utilities.densities import laplace_pdf

b_1 = 1.
b_2 = 2.
mu = 1.
bounds = -5, 5
cont_support = np.linspace(*bounds, num=1000)

pdf_1 = laplace_pdf(b_1, mu=mu)
plt.plot(cont_support, pdf_1(cont_support), label=f"$Y \sim \mathrm{{Lap}}(\mu={mu}, b={b_1})$")

pdf_2 = laplace_pdf(b_2, mu=mu)
plt.plot(cont_support, pdf_2(cont_support), label=f"$Y \sim \mathrm{{Lap}}(\mu={mu}, b={b_2})$")

plt.plot([mu, mu], [0, pdf_1(mu)], linestyle='dashed', color="black")
plt.text(mu + .1, 0.01, f"$\mu = {mu}$")

plt.ylim(bottom=0)
plt.xlim(*bounds)
plt.ylabel(f"$\Pr[Y = y]$")
plt.xlabel("y")
plt.legend()
plt.title("Laplace distribution for specific parameterizations of $\mu$ and $b$")
plt.savefig("../images/ch01_laplace.png")
# plt.show()
