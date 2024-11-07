import opendp.prelude as dp
import numpy as np
from ch06_partitioning import make_partition_randomly


def make_sample_and_aggregate(
      input_domain, output_domain, num_partitions, black_box):
    dp.assert_features("contrib", "honest-but-curious")

    # SAMPLE: randomly partition data into `num_partitions` data sets
    trans_subsample = make_partition_randomly(input_domain, num_partitions)

    # AGGREGATE: apply `black_box` function to each partition
    trans_aggregate = dp.t.make_user_transformation(
        input_domain=dp.vector_domain(input_domain, size=num_partitions),
        output_domain=dp.vector_domain(output_domain, size=num_partitions),
        input_metric=dp.symmetric_distance(),
        output_metric=dp.hamming_distance(),
        function=lambda parts: [black_box(part) for part in parts],
        stability_map=lambda b_in: b_in) # 1-stable

    return trans_subsample >> trans_aggregate


if __name__ == "__main__":
    # sample and aggregate
    from sklearn.metrics import f1_score
    meas_sa = make_sample_and_aggregate(
        dp.np_array2_domain(T=float), dp.atom_domain(T=float),
        num_partitions=50,
        black_box=lambda data: f1_score(data[:, 0], data[:, 1]),
    )

    data = np.random.normal(size=(10_000, 2))
    meas_sa(data)

