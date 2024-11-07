import opendp.prelude as dp
import numpy as np
from ch03_np_clamp import make_np_clamp
from ch03_np_sum import make_np_sum
from ch06_partitioning import make_partition_randomly, partition_distance
from ch05_advanced_composition import get_elements


def make_parallel_composition(mechanisms):
    dp.assert_features("contrib")

    input_domain, input_metric, output_measure = get_elements(mechanisms)

    assert input_metric == dp.symmetric_distance(), "expected microdata input"

    def privacy_map(b_in):
        l0, _l1, linf = b_in
        return l0 * max(m.map(linf) for m in mechanisms)

    return dp.m.make_user_measurement(
        input_domain=dp.vector_domain(input_domain, size=len(mechanisms)), 
        input_metric=partition_distance(dp.symmetric_distance()), 
        output_measure=output_measure,
        # apply the ith mechanism to the ith partition
        function=lambda parts: [m_i(p_i) for m_i, p_i in zip(mechanisms, parts)],
        # privacy loss is the max among partitions
        privacy_map=privacy_map)


if __name__ == "__main__":
    dp.enable_features("contrib", "honest-but-curious", "floating-point")

    mock_data = np.random.normal(size=(1000, 4))
    n_parts = 5

    # noninteractive parallel composition
    meas_sum = make_np_clamp(norm=4., p=1) >> \
               make_np_sum(norm=4., p=1) >> \
               dp.m.then_laplace(scale=4.)
    
    partitioner = make_partition_randomly(dp.np_array2_domain(T=float), n_parts)
    meas_pc = partitioner >> make_parallel_composition([meas_sum] * 5)

    print(meas_pc(mock_data))
