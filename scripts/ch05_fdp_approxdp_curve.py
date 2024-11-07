import numpy as np

def f_approxDP(profile):
    def curve(alpha):
        return max(max(
            0,
            1 - delta - np.exp(epsilon) * alpha,
            np.exp(-epsilon) * (1 - delta - alpha),
        ) for epsilon, delta in profile)
    return curve

curve = f_approxDP([(1., 1e-6)])
alpha = .2
beta = curve(alpha)
print(alpha, beta)