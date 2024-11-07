from ch04_stream import make_above_threshold, query_type, queryable_domain
import opendp.prelude as dp


def make_sparse_vector(threshold, scale, k, monotonic=False):
    """Privately find the first k items above `threshold` in a stream."""
    dp.assert_features("contrib", "floating-point")

    meas_AT = make_above_threshold(threshold, scale, monotonic)
    meas_BC = dp.c.make_basic_composition([meas_AT] * k)

    def f_sparse_vector(stream):
        qbls_AT, found = meas_BC(stream), 0
        def transition(query):
            nonlocal found
            assert found < k, "sparse vector mechanism is exhausted"
            if qbls_AT[found](query):
                found += 1
                return True
            return False
        
        return dp.new_queryable(transition, Q=query_type(stream), A=bool)
        
    return dp.m.make_user_measurement(
        input_domain=queryable_domain(dp.atom_domain(T=type(threshold))),
        input_metric=dp.linf_distance(T=type(threshold), monotonic=monotonic),
        output_measure=dp.max_divergence(T=float),
        function=f_sparse_vector,
        privacy_map=lambda b_in: b_in * meas_BC.map(1),
    )


def test_sparse_vector():
    dp.enable_features("honest-but-curious", "contrib", "floating-point")
    meas = make_sparse_vector(threshold=4, scale=1., k=3)
    # since the measurement privatizes a stream, initialize with a dummy stream
    qbl = meas(lambda x: x)

    qbl(1) # very likely to emit False
    qbl(1) # likely to emit False
    qbl(8) # likely to emit True for the first time
    qbl(1) # likely to emit False
    qbl(8) # likely to emit True for the second time
    qbl(8) # likely to emit True for the third time
    qbl(7) # expected to throw an assertion error

test_sparse_vector()