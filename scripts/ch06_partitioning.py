import opendp.prelude as dp
import numpy as np

from ch03_d_sym import d_Sym


# Lp Sensitivity
# 1. group               -> |Sym|_0, |Sym|_1, |Sym|_∞
# 2. stable agg trans    -> Lp distance

# Parallel Composition
# 1. group or partition  -> |Sym|_0, |Sym|_∞
# 2. measurement         -> privacy measure

# Subsample/Aggregate
# 1. group or partition  -> |Sym|_0
# 2. black-box aggregate -> Sym

# partition: l0 = linf = b_in
# group:     l0 = descriptor or b_in; linf = descriptor or b_in

# |d_M(x_i, x'_i)|_0 for any metric M is equivalent to hamming(x, x')


def dataframe_domain(grouped_l0_li=None):
    return dp.user_domain(
        "DataframeDomain()", 
        member=lambda x: isinstance(x, pd.DataFrame), 
        descriptor=grouped_l0_li
    )


def series_domain():
    return dp.user_domain("SeriesDomain()", member=lambda x: isinstance(x, pd.Series))


def groupby_domain():
    return dp.user_domain("GroupByDomain()", member=lambda _: True)


def d_Partition(x, x_p, d=d_Sym):
    """L0, L1 and L∞ norms of the distances between neighboring partitions"""
    dists = abs(np.array([d(x_i, y_i) for x_i, y_i in zip(x, x_p)]))
    #      |d(x, x')|_0     , |d(x, x')|_1, |d(x, x')|_∞
    return (dists != 0).sum(), dists.sum(), dists.max()


def partition_distance(inner_metric=dp.symmetric_distance()):
    """Distance metric for partitioned datasets using d_Partition"""
    return dp.user_distance(f"PartitionDistance({inner_metric})")


def can_be_partitioned(domain):
    # not implemented
    _ = domain
    return True


def make_partition_randomly(input_domain, num_partitions):
    dp.assert_features("contrib")
    assert can_be_partitioned(input_domain) # like vectors, arrays

    def f_partition_randomly(data):
        data = np.array(data, copy=True)
        # randomly partition data into `num_partitions` data sets
        np.random.shuffle(data)
        return np.array_split(data, num_partitions)

    return dp.t.make_user_transformation(
        input_domain=input_domain,
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(input_domain, size=num_partitions),
        output_metric=partition_distance(dp.symmetric_distance()),
        function=f_partition_randomly,
        # TODO: what kind of domain descriptors can you use to improve this?
        # TODO: how might you modify the function to improve this?
        stability_map=lambda b_in: (b_in, b_in, b_in))


