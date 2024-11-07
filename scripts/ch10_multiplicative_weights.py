import numpy as np

def get_mask(x):
    # not implemented
    _ = x

def multiplicative_weights(A, selected_qs, released_ys):
    n = A.sum()
    old_A = np.zeros_like(A)
    while not np.allclose(A, old_A): # run until convergence
        for q_i, y_i in zip(selected_qs, released_ys):
            error = y_i - q_i(A)
            M_i = get_mask(q_i) # an array of zeroes or ones in the shape of A

            # multiplicative weights update
            A *= np.exp(M_i * error / (2 * n))
            # re-normalize so that the same number of records remain in the data
            A *= n / A.sum()
        old_A = A
    return A
