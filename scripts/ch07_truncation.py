import opendp.prelude as dp
import pandas as pd
import numpy as np
from ch06_partitioning import dataframe_domain, groupby_domain, partition_distance
from ch07_identifier_distance import id_distance

INF = float("inf")


def make_truncate(id_col, threshold):
    """"
    make a truncation transformation that samples `threshold` records per id
    """
    dp.assert_features("contrib")
    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=id_distance(id_col),
        output_domain=dataframe_domain(),
        output_metric=dp.symmetric_distance(),
        function=lambda df: df.groupby(id_col).sample(n=threshold, replace=True),
        stability_map=lambda b_in: b_in * threshold)


def make_group_by_truncate(id_col, by, l0=INF, l1=INF, li=INF):
    dp.assert_features("contrib")

    def f_bound_the_contribution(df: pd.DataFrame):
        if l0 != INF:  # limit the number of partitions an id can influence
            contribs = df.groupby([id_col, *by]).size()
            kept_groups = contribs.groupby(id_col, group_keys=False).nlargest(l0)
            df = df.merge(kept_groups.index.to_frame(index=False))

        if l1 != INF:  # limit the total number of records an id can have
            df = df.groupby(by=id_col).sample(n=l1, replace=True)

        if li != INF:  # limit the records an id can have per-partition
            df = df.groupby([id_col, *by]).sample(n=li, replace=True)

        # return grouped data with bounded l0, l1, and/or linfty contribution
        return df.groupby(by=by)
    # ...

    # ...
    # tighten when possible
    l0_l1_li_p = np.array([min(l0, l1), min(l1, l0 * li), min(li, l1)])
    
    return dp.t.make_user_transformation(
        input_domain=dataframe_domain(),
        input_metric=id_distance(id_col),
        output_domain=groupby_domain(),
        output_metric=partition_distance(dp.symmetric_distance()),
        function=f_bound_the_contribution,
        stability_map=lambda b_in: b_in * l0_l1_li_p)
