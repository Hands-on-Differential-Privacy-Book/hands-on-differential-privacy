import matplotlib.pyplot as plt
import numpy as np
from utilities.densities import gaussian_pdf

scale = 1.
b_in = 1
gpdf = gaussian_pdf(scale)
cont_support = np.linspace(-5, 5, num=1000)
plt.plot(cont_support, gpdf(cont_support), label="$M(\cdot) \sim \\mathrm{Normal}(0)$")
plt.plot(cont_support, gpdf(cont_support + b_in), label="$M(\cdot) \sim \\mathrm{Normal}(b_{in})$")

def plot_vline(x):
    p_x = gpdf(x)
    p_x_p = gpdf(x + b_in)
    loss = np.log(p_x / p_x_p)
    plt.plot([x, x], [p_x, p_x_p], color="red")
    plt.text(x, p_x, f"$\mathcal{{L}}$ = {loss}")

for line in [1.5, 2, 2.5, 3.0]:
    plot_vline(line)

plt.xlabel("support (y)")
plt.ylabel("$\Pr[M(\cdot) = y]$")
plt.title("Distribution of Gaussian Mechanism on Adjacent Data Sets")
plt.legend()
plt.savefig("../images/ch05_gaussian_log_ratio.png")
plt.show()

