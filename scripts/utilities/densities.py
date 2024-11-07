import numpy as np

def laplace_pdf(b, mu=0):
    return lambda x: np.exp(-abs(x - mu) / b) / (2 * b)

def gaussian_pdf(b, mu=0):
    return lambda x: np.exp(-((x - mu) / b) ** 2 / 2) / (np.sqrt(2 * np.pi) * b)

def geometric_pdf(b, mu=0):
    return lambda x: (np.exp(1/b) - 1) / (np.exp(1/b) + 1) * np.exp(-abs(x - mu) / b)

def truncated_laplace_pdf(scale, radius, truncate=True):
    def func(x):
        B = 1 / (2 * scale * (1 - np.exp(-radius / scale)))
        densities = B * np.exp(-abs(x) / scale)
        if truncate:
            densities[abs(x) > radius] = 0
        return densities
    return func
