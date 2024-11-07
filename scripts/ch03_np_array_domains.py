import numpy as np
import opendp.prelude as dp


# the domain of all 2-dimensional arrays with 4 columns
domain = dp.np_array2_domain(num_columns=4, T=float)

# arrays with 4 columns are in this domain, but 3 columns are not
assert domain.member(np.random.normal(size=(1_000, 4)))
assert not domain.member(np.random.normal(size=(1_000, 3)))


# the domain of all 2-dimensional arrays with row 1-norm at most 4
domain = dp.np_array2_domain(norm=4., p=1, origin=[0.] * 4)

# this data set is in the domain because the L1-norm of each row is at most 4
assert domain.member(np.array([
    [4.0,  0.0,  0.0, 0.0], # row 1 with L1-norm of 4.0
    [1.1, -2.7,  0.0, 0.0], # row 2 with L1-norm of 3.8
    [0.9,  0.2, -2.1, 0.5], # row 3 with L1-norm of 3.7
]))