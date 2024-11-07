import opendp.prelude as dp
import numpy as np
from ch06_partitioning import dataframe_domain, groupby_domain, partition_distance, series_domain


def extract_prior_knowledge(domain, by):
    # NOT IMPLEMENTED
    _ = domain, by
    return 4, 1


def make_groupby(input_domain, by):
    dp.assert_features("contrib")
    # prior knowledge about l0 and l∞ norm are inherent to the data frame's domain
    l0, linf = extract_prior_knowledge(input_domain, by)
    
    return dp.t.make_user_transformation(
        input_domain=input_domain,
        input_metric=partition_distance(),
        output_domain=groupby_domain(),
        output_metric=partition_distance(),
        function=lambda df: df.groupby(by),
        stability_map=lambda b_in: (min(l0, b_in), b_in, min(linf, b_in)))


def make_group_size():
    """make a transformation that computes the size of each group"""
    dp.assert_features("contrib")
    return dp.t.make_user_transformation(
        input_domain=groupby_domain(),
        input_metric=partition_distance(),
        output_domain=series_domain(),
        output_metric=dp.l2_distance(T=float),
        function=lambda groupby: groupby.size(),
        stability_map=lambda l0_l1_li: np.sqrt(l0_l1_li[0]) * l0_l1_li[2])


def make_pd_gaussian(scale):
    """Make a gaussian noise mechanism that privatizes a series.
    Assumes the index is public information."""
    dp.assert_features("contrib")

    space = dp.vector_domain(dp.atom_domain(T=float)), dp.l2_distance(T=float)
    m_gauss = dp.m.make_gaussian(*space, scale)

    return dp.m.make_user_measurement(
        input_domain=series_domain(),  # assuming that the index is public info
        input_metric=m_gauss.input_domain,
        output_measure=m_gauss.output_measure,
        function=lambda s: pd.Series(m_gauss(s.to_numpy()), index=s.index),
        privacy_map=m_gauss.map)


if __name__ == "__main__":
    import numpy as np
    import pandas as pd

    dp.enable_features("contrib", "honest-but-curious")

    n_rows = 10_000
    df = pd.DataFrame(
        {
            "Home Country": np.random.choice(["Nigeria", "Bangladesh", "Mexico"], size=n_rows),
            "Year": np.random.choice(["2019", "2020", "2021"], size=n_rows),
        }
    )


    input_domain = dataframe_domain(
        # when grouped by "Home Country"
        # an individual may contribute to at most one partition
        max_partitions={"Home Country": 1},
        # when grouped by "Year"
        # each partition has at most one contribution from each individual
        max_per_partition={"Year": 1})
    
    m_gb = (
        make_groupby(input_domain, by=["Home Country", "Year"])
        >> make_group_size()
        >> make_pd_gaussian(scale=1.0)
    )

    print(m_gb(df))
    print(m_gb.map(1))
