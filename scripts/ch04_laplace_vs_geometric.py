import numpy as np
import matplotlib.pyplot as plt
from utilities.densities import laplace_pdf, geometric_pdf

scale = 1.
cont_support = np.linspace(-10, 10, num=1000)
plt.plot(cont_support, laplace_pdf(scale)(cont_support), label="$M(\cdot)$ = Lap(0)")
cont_support = np.arange(-10, 11)
plt.step(cont_support, geometric_pdf(scale)(cont_support), where='mid', label="$M(\cdot)$ = DLap(0)")
plt.xlabel("support (x)")
plt.ylabel("$\Pr[M(\cdot) = x]$")
plt.title("Laplace Mechanism vs. Discrete Laplace (Geometric) Mechanism")
plt.legend()
plt.savefig("../images/ch04_laplace_vs_geometric.png")
# plt.show()

