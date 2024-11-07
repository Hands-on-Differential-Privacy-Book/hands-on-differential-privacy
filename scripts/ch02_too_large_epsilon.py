import matplotlib.pyplot as plt
import numpy as np
eps = np.linspace(0, 5, num=100)
plt.plot(eps, np.exp(eps))
plt.xlabel("epsilon ($\epsilon$)")
plt.ylabel("odds ratio")
plt.savefig("../images/ch02_too_large_epsilon.png")
plt.show()
