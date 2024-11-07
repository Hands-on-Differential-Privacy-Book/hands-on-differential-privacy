from ch04_interval_exponential_mechanism import make_private_quantile_in_bounds
import opendp.prelude as dp


def make_private_bounds_via_quantile(loose_bounds, scale):
    return dp.c.make_basic_composition([
        make_private_quantile_in_bounds(loose_bounds, alpha=0.05, scale=scale),
        make_private_quantile_in_bounds(loose_bounds, alpha=0.95, scale=scale),
    ])
