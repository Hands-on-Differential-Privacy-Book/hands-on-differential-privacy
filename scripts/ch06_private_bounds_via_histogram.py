import opendp.prelude as dp
import numpy as np
dp.enable_features("contrib", "honest-but-curious")


def make_logarithmic_binning():
    """bin each datum into a floating-point band"""
    dp.assert_features("contrib")
    input_domain = dp.vector_domain(dp.atom_domain(T=float))
    output_domain = dp.vector_domain(dp.atom_domain(T=int))
    metric = dp.symmetric_distance()

    def f_bin_by_band(arg):
        arg = np.array(arg)
        band = np.log2(abs(arg)) # bands grow exponentially
        band[~np.isfinite(arg)] = 1025 # infinities get their own band
        return ((arg >= 0) * 2 - 1) * (band.astype(int) + 1074) # sign * band

    return dp.t.make_user_transformation(
        input_domain, metric, output_domain, metric, 
        f_bin_by_band, lambda b_in: b_in)  # 1-stable

def make_find_bounds(scale, alpha=1e-9):
    """makes a postprocessor that finds bounds from a vector of noisy counts"""
    n_bands = 2099 + 1 + 2099 # negative bands, zero, positive bands
    threshold = -scale * np.log(2 - 2 * (1 - alpha) ** (1 / (n_bands - 1)))

    def f_find_bounds_from_bin_counts(counts):
        assert len(counts) == n_bands, "expected one count per-band"
        lower_idx = (counts > threshold).argmax() - 1
        upper_idx = n_bands - (counts > threshold)[::-1].argmax()
        idx = np.array([lower_idx, upper_idx]) - 2099

        with np.errstate(over="ignore"):
            return ((idx >= 0) * 2 - 1) * 2. ** (abs(idx) - 1075)

    return f_find_bounds_from_bin_counts


def make_private_bounds_via_histogram(scale, alpha=1e-9):
    return (
        make_logarithmic_binning() >> 
        dp.t.then_count_by_categories(np.arange(-2099, 2100), False) >>
        dp.m.then_laplace(scale) >>
        make_find_bounds(scale, alpha)
    )


def test_binning():
    binner = make_logarithmic_binning()
    print(binner([np.nextafter(0, 1), 2.2250738585072014e-308, 0.5, 1., 2., 1.7976931348623157e+308, np.inf]))
    print(binner([np.nextafter(0, -1), -2.2250738585072014e-308, -0.5, -1., -2., -1.7976931348623157e+308, -np.inf]))


# meas = make_private_bounds_via_histogram(scale)
# print(meas(np.random.normal(size=1000)))

def test_make_find_bounds():
    post = make_find_bounds(scale=1.)
    zeros = np.zeros(2099 + 1 + 2099)
    # zeros[0] = 1000
    # zeros[1] = 1000
    zeros[2100 + 1074] = 1000
    print(post(zeros))

meas = make_private_bounds_via_histogram(scale=3., alpha=1e-9)
print(meas(np.random.normal(size=1000, scale=10))) # ~> [-8.0, 8.0]
print(meas.map(1)) # -> .333 = ε