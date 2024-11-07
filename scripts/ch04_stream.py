import opendp.prelude as dp
from opendp.core import queryable_query_type
from numpy.random import laplace


def query_type(queryable):
    if isinstance(queryable, dp.Queryable):
        return queryable_query_type(queryable.value)
    return "ExtrinsicObject"


def queryable_domain(output_domain):
    """Describes the set of all streams emitting values in `output_domain`"""
    return dp.user_domain(
        f"QueryableDomain(output_domain={str(output_domain)})",
        member=lambda x: isinstance(x, dp.Queryable), 
        descriptor={"output_domain": output_domain})

    
def make_above_threshold(threshold, scale, monotonic=False):
    """Privately find the first item above `threshold` in a stream."""
    dp.assert_features("contrib", "floating-point")

    def f_above_threshold(stream):
        found, threshold_p = False, laplace(loc=threshold, scale=scale)
        def transition(query):
            nonlocal found # the state of the queryable
            assert not found, "queryable is exhausted"

            value = laplace(loc=stream(query), scale=2 * scale)
            if value >= threshold_p:
                found = True
            return found
        return dp.new_queryable(transition, Q=query_type(stream), A=bool)
        
    return dp.m.make_user_measurement(
        input_domain=queryable_domain(dp.atom_domain(T=type(threshold))),
        input_metric=dp.linf_distance(T=type(threshold), monotonic=monotonic),
        output_measure=dp.max_divergence(T=float),
        function=f_above_threshold,
        privacy_map=lambda b_in: b_in / scale * (1 if monotonic else 2),
    )


def make_query_stream(input_domain, input_metric, b_in, b_out):
    """Return a stream for asking queries about a data set"""
    dp.assert_features("contrib")
    T = type(b_out)
    input_space = input_domain, input_metric
    output_space = dp.atom_domain(T=T), dp.absolute_distance(T=T)
    stream_space = queryable_domain(output_space[0]), dp.linf_distance(T=T)

    def f_query_stream(data):
        def transition(trans):
            assert trans.input_space == input_space
            assert trans.output_space == output_space
            assert trans.map(b_in) <= b_out, "query is too sensitive"
            return trans(data)
        return dp.new_queryable(transition, A=T)

    return dp.t.make_user_transformation(
        *input_space, *stream_space, f_query_stream,
        stability_map=lambda b_in_p: b_in_p / b_in * b_out
    )
