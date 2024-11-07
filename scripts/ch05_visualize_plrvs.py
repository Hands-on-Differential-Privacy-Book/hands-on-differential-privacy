import numpy as np
import matplotlib.pyplot as plt
from utilities.densities import laplace_pdf

mu = 10.
scale = 1.
sensitivity = 1.
support = np.linspace(5, 15, num=1000)

def plrv(pdf):
    return lambda x: np.log(pdf(x) / pdf(x - sensitivity))

outcomes = [7, 10.5, 13]
p1_y = [.08, .01, .08]
pdf_lap = laplace_pdf(scale, mu=mu)

fig, ax = plt.subplots(1, 2)


pdf_1 = laplace_pdf(scale, mu=mu)
pdf_2 = laplace_pdf(scale, mu=mu)

ax[0].plot(support, pdf_1(support), label=f"$\Pr[M(x) = y]$")
ax[0].plot(support, pdf_2(support - sensitivity), label=f"$\Pr[M(x') = y]$")
ax[0].legend()
for i, y in enumerate(outcomes):
    ax[0].plot([y, y], [-.01, 0.5], linestyle='dashed', color="black")
    ax[0].text(y + .1, -.02, f"$y_{i} = {y}$")
ax[0].set_xlabel("support (denoted $y$)")
ax[0].set_ylabel("$\Pr[Y = y]$")
ax[0].set_title("Laplace mechanism distributions on adjacent datasets")

ax[1].plot(support, plrv(pdf_lap)(support), label="Laplace")
for i, y in enumerate(outcomes):
    ax[1].plot([y, y], [-1.05, 1], linestyle='dashed', color="black")
    ax[1].text(y + .1, -1.1, f"$y_{i} = {y}$")
# ax[1].legend()
ax[1].set_xlabel("support (denoted $y$)")
ax[1].set_ylabel("$\mathcal{L}_{M(x) / M(x')}(y)$")
ax[1].set_title("Privacy loss across outcomes")

fig.set_figwidth(10)
fig.set_figheight(4)
plt.savefig("../images/ch05_laplace_plrv.png")
plt.show()