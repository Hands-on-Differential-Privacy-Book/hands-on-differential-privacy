import opendp.prelude as dp
dp.enable_features("contrib", "honest-but-curious")


m_rr = dp.m.make_randomized_response_bool(prob=.55)
m_sc = dp.c.make_sequential_composition(
    m_rr.input_domain, m_rr.input_metric, m_rr.output_measure,
    1, [1., 1.]
)
# spawns an instance of the compositor that you can interact with
qbl = m_sc(True)
# consume the first privacy parameter (of 2) by releasing a randomized response
qbl(m_rr) # ~> True

# The compositor now has one remaining query that may consume ε = 1.
# Can nest a second compositor to sub-divide the remaining budget:
m_sc2 = dp.c.make_sequential_composition(
    m_rr.input_domain, m_rr.input_metric, m_rr.output_measure,
    1, [0.6, 0.4]
)
qbl2 = qbl(m_sc2)
# the sub-compositor is now exhausted:
qbl2(m_rr) # ~> True
qbl2(m_rr) # ~> False


def make_sequential_composition(
    input_domain, input_metric, output_measure, b_in, b_mids):
    """When invoked with some data, spawns a compositor queryable."""
    dp.assert_features("contrib")
    # this example implementation assumes the privacy measure is pure-DP
    assert output_measure == dp.max_divergence(T=float)

    # when data is passed to the measurement...
    def f_data_to_compositor_queryable(data):
        # ...a new queryable is spawned that tracks the privacy expenditure
        d_mids_p = list(b_mids)

        # this function will be called each time a query is passed
        def transition(query):
            # ensure that the query (a measurement) can be applied to the data
            assert query.input_domain == input_domain
            assert query.input_metric == input_metric
            assert query.output_measure == output_measure
            assert query.map(b_in) <= d_mids_p[0], "insufficient budget for query"

            release = query(data)
            d_mids_p.pop(0) # consume the budget if the release is successful
            return release

        return dp.new_queryable(transition)

    def privacy_map(b_in_p):
        assert b_in_p <= b_in
        return b_in * sum(b_mids)

    return dp.m.make_user_measurement(
        input_domain, input_metric, output_measure,
        function=f_data_to_compositor_queryable, 
        privacy_map=privacy_map)
