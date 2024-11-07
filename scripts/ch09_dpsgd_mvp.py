from ch03_np_clamp import make_np_clamp
from ch03_np_sum import make_np_sum

import opendp.prelude as dp
import numpy as np


dp.enable_features("floating-point", "contrib", "honest-but-curious")


def make_nabla_loss_i(w):
    dp.assert_features("contrib", "floating-point")
    w_0, w_1 = w
    def f_compute_grads(data):
        x, y = data[np.newaxis].T
        y_hat = w_0 + w_1 * x  # forward pass y^ = f(x)
        return (y_hat - y) * np.column_stack([np.ones(x.size), x])

    space = dp.np_array2_domain(T=float), dp.symmetric_distance()
    return dp.t.make_user_transformation(
        *space, *space, f_compute_grads,
        stability_map=lambda b_in: b_in)


N = 100_000  # public metadata

# "load" the data
x = np.random.uniform(-5, 5, size=N)
y = 3 + 2 * x + np.random.normal(size=x.size)
data = np.column_stack((x, y))
max_contributions = 1

# model hyperparameters
w = np.array([0.0, 0.0]) # initial choice of params
gamma, num_steps = 0.3, 20
norm = 2. # assumes most grads have magnitude lte 2
noise_std = 100.


sum_meas = make_np_clamp(norm, p=2) >> \
           make_np_sum(norm, p=2) >> \
           dp.m.then_gaussian(scale=noise_std) >> \
           np.array # a postprocessor- load into a numpy array

meas_comp = dp.c.make_sequential_composition(
    input_domain=sum_meas.input_domain,
    input_metric=sum_meas.input_metric,
    output_measure=dp.zero_concentrated_divergence(T=float),
    d_in=max_contributions,
    d_mids=[sum_meas.map(max_contributions)] * num_steps
)
# qbl is an instance of the compositor that allows up to `num_steps` queries
qbl = meas_comp(data)
# now the only way to access the data is through the compositor
del data

print(meas_comp.map(max_contributions)) # -> 0.004 = ρ
εδ_curve = dp.c.make_zCDP_to_approxDP(meas_comp).map(max_contributions)
print(εδ_curve.epsilon(1e-8))           # -> (0.4659, 1e-8) = (ε, δ)

# train
for _ in range(num_steps):
    # make a mechanism that computes the gradient
    meas_nabla_loss = make_nabla_loss_i(w) >> sum_meas
    # privately release the gradient by querying the compositor
    w -= gamma * 2 / N * qbl(meas_nabla_loss)

# assess utility
print("params:", w) # ~> [3.00183246 1.97430499]


# import matplotlib.pyplot as plt
# deltas = np.linspace(0, 1e-4, num=100)
# epsilons = [εδ_curve.epsilon(d) for d in deltas]
# plt.plot(deltas, epsilons)
# plt.show()

from ch05_rdp_to_fixed_approx_dp import renyi_divergence
from ch06_odometer import make_sequential_odometer, Map
from ch09_sampled_gaussian import make_sampled_gaussian
x = np.random.uniform(-5, 5, size=N)
y = 3 + 2 * x + np.random.normal(size=x.size)
data = np.column_stack((x, y))
w = np.array([0.0, 0.0]) # initial choice of params

q = 1e-2 # sampling rate

sum_trans = make_np_clamp(norm, p=2) >> make_np_sum(norm, p=2)
odometer = make_sequential_odometer(
    input_domain=sum_trans.input_domain,
    input_metric=sum_trans.input_metric,
    output_measure=renyi_divergence()
)
qbl = odometer(data)

# train
for _ in range(num_steps):
    # make a mechanism that computes the gradient
    trans_nabla_loss = make_nabla_loss_i(w) >> sum_trans
    meas_nabla_loss = make_sampled_gaussian(trans_nabla_loss, noise_std, q)
    # privately release the gradient by querying the compositor
    w -= gamma * 2 / (N * q) * qbl(meas_nabla_loss)

rdp_curve = qbl(Map(b_in=max_contributions))


from math import log
# check likely alphas and return the best ε
delta = 1e-8
epsilon = min(rdp_curve(a) + log(1 / delta) / (a - 1) for a in range(2, 300))

print((epsilon, delta)) # -> (.0619, 1e-8) = (ε, δ)
print("params:", w) # ~> [3.09856293 2.06650036]

