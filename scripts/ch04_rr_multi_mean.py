from ch04_rr_multi import make_randomized_response_multi
import opendp.prelude as dp

dp.enable_features("honest-but-curious", "contrib", "floating-point")

from random import uniform, randint
from statistics import mean

def make_randomized_round(bounds):
    dp.assert_features("contrib")
    domain, metric = dp.atom_domain(T=type(bounds[0])), dp.discrete_distance()
    return dp.t.make_user_transformation(
        domain, metric, domain, metric,
        function=lambda arg: bounds[int(uniform(*bounds) < arg)],
        stability_map=lambda b_in: b_in
    )

bounds = [0, 100]
rr_mean = make_randomized_round(bounds) \
          >> make_randomized_response_multi(.75, bounds)

# if you know the data is actually distributed evenly between bounds,
# the estimator is unbiased
mock_dataset = [randint(*bounds) for _ in range(10_000)]
priv_dataset = [rr_mean(x) for x in mock_dataset]

print(mean(mock_dataset))
print(mean(priv_dataset))