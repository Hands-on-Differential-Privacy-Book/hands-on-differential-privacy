import matplotlib.pyplot as plt
import numpy as np
from utilities.densities import laplace_pdf, gaussian_pdf

scale = 1.
cont_support = np.linspace(-5, 5, num=1000)
plt.plot(cont_support, laplace_pdf(scale)(cont_support), label="$M(\cdot)$ is Laplace(0)")
plt.plot(cont_support, gaussian_pdf(scale)(cont_support), label="$M(\cdot)$ is Normal(0)")
plt.xlabel("support (y)")
plt.ylabel("$\Pr[M(\cdot) = y]$")
plt.title("Distribution of Laplace Mechanism vs. Gaussian Mechanism")
plt.legend()
plt.savefig("../images/ch05_laplace_vs_gaussian.png")
plt.show()

