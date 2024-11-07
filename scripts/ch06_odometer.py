from dataclasses import dataclass
from typing import Any, Callable
import opendp.prelude as dp
from opendp.core import function_eval
from ch05_rdp_to_fixed_approx_dp import renyi_divergence

# This file contains a prototype odometer implementation
#    for use in explaining odometers in the Hands-on-Differential-Privacy book
# It has known issues, but exposes the expected API
# The filter's continuation rule doesn't fire when it should
# For book purposes, this is fine- the comments explain when it should fire


@dataclass
class Odometer(object):
    input_domain: dp.Domain
    function: Callable[[Any], Any]
    input_metric: dp.Metric
    output_measure: dp.Measure

    def __call__(self, arg):
        return self.function(arg)


@dataclass
class ChildChange(object):
    pending_map: Callable[[Any], Any]
    id: int


@dataclass
class Map(object):
    b_in: float


@dataclass
class MapAfter(object):
    proposed_query: Any


@dataclass
class GetId(object):
    pass


WRAPPER: Callable[[dp.Queryable], dp.Queryable] = None


def with_wrapper(new_wrapper: Callable[[dp.Queryable], dp.Queryable], f: Callable[[], Any]):
    global WRAPPER
    prev_wrapper = WRAPPER

    if prev_wrapper is None:
        WRAPPER = new_wrapper
    else:
        def chained_wrapper(q): 
            return prev_wrapper(new_wrapper(q))
        WRAPPER = chained_wrapper

    try:
        answer = f()
    finally:
        WRAPPER = prev_wrapper

    return answer


# BEGIN WRAPPER HELPERS
def _new_getid_wrapper(id):
    """Adds a wrapper queryable that reports its id when queried"""
    def wrap_logic(inner_queryable) -> dp.Queryable:
        def getid_wrapper_transition(query):
            if isinstance(query, GetId):
                return id
            # the inner queryable may handle any other kind of query
            return inner_queryable.eval(query)
        return dp.new_queryable(getid_wrapper_transition)
    return wrap_logic


def _new_filter_wrapper(parent_queryable):
    """Constructs a function that recursively wraps child queryables.
    All queryable descendants recursively report privacy usage to their parents"""
    def wrap_logic(inner_queryable) -> dp.Queryable:
        wrapper_qbl = None
        def filter_wrapper_transition(query):
            nonlocal wrapper_qbl
            if isinstance(query, (dp.Measurement, Odometer)):
                # Determine privacy usage after the proposed query
                pending_privacy_map = inner_queryable.eval(
                    MapAfter(query))
                
                # will throw an exception if the privacy budget is exceeded
                parent_queryable.eval(ChildChange(
                    pending_map=pending_privacy_map,
                    id=inner_queryable.eval(GetId())
                ))

                # recursively wrap any child queryable in the same logic,
                #     but in a way that speaks to this wrapper instead
                return with_wrapper(
                    lambda qbl: _new_filter_wrapper(wrapper_qbl)(qbl),
                    lambda: inner_queryable.eval(query))

            # if a child queryable reports a potential change in its privacy usage,
            #     work out the new privacy consumption at this level,
            #     then report it to the parent queryable
            if isinstance(query, ChildChange):
                pending_map = inner_queryable.eval(query)
                parent_queryable.eval(ChildChange(
                    id=inner_queryable.eval(GetId()),
                    pending_map=pending_map
                ))
                return pending_map

            # the inner queryable may handle any other kind of query
            return inner_queryable.eval(query)

        wrapper_qbl = dp.new_queryable(filter_wrapper_transition)
        return wrapper_qbl
    return wrap_logic


def make_sequential_odometer(input_domain, input_metric, output_measure):
    """Make an odometer that spawns a concurrent odometer queryable when invoked with some data."""
    def function(data):
        # queryable state
        child_maps = []

        def transition(query):
            nonlocal data, child_maps, input_domain, input_metric, output_measure

            if isinstance(query, dp.Measurement):
                assert input_domain == query.input_domain
                assert input_metric == query.input_metric
                assert output_measure == query.output_measure
                child_maps.append(query.map)
                child_id = len(child_maps)

                getid_wrapper = _new_getid_wrapper(child_id)
                answer = with_wrapper(
                    getid_wrapper, lambda: function_eval(query.function, data, TI=query.input_carrier_type))
                return answer

            if isinstance(query, Map):
                d_mids = [child_map(query.b_in) for child_map in child_maps]
                if output_measure == renyi_divergence():
                    return lambda alpha: sum(curve(alpha) for curve in d_mids)
                if output_measure == dp.max_divergence(T=float):
                    return sum(d_mids)
                raise NotImplementedError(f"composition not implemented for {output_measure}")

            # This is needed to answer queries about the hypothetical privacy consumption after running a query.
            # This gives the filter a way to reject the query before the state is changed.
            if isinstance(query, MapAfter):
                if isinstance(query.proposed_query, dp.Measurement):
                    def pending_map(b_in):
                        pending_maps = [*child_maps,
                                        query.proposed_query.privacy_map]
                        d_mids = [child_map(b_in)
                                  for child_map in pending_maps]
                        return output_measure.compose(d_mids)
                    return pending_map
            raise ValueError(f"unrecognized query: {query}")

        return dp.new_queryable(transition)

    return Odometer(
        input_domain=input_domain,
        input_metric=input_metric,
        output_measure=output_measure,
        function=function
    )



def make_odometer_to_filter(odometer, b_in, b_out):
    """Construct a filter measurement out of an odometer.
    Limits the privacy consumption of the queryable that the odometer spawns."""
    def function(data):
        nonlocal odometer, b_in, b_out

        # construct a filter queryable
        def filter_transition(query):
            nonlocal b_in, b_out

            if isinstance(query, ChildChange):
                if query.pending_map(b_in) > b_out:
                    raise ValueError("privacy budget exceeded")
                return query.pending_map
            raise ValueError("unrecognized query", query)
        filter_queryable = dp.new_queryable(filter_transition)

        # the child odometer always identifies itself as id 0
        getid_wrapper = _new_getid_wrapper(0)
        recursive_wrapper = _new_filter_wrapper(filter_queryable)
        def wrapper(q): 
            return recursive_wrapper(getid_wrapper(q))

        return with_wrapper(wrapper, lambda: function_eval(odometer.function, data, TI=odometer.input_domain.carrier_type))

    def privacy_map(b_in_prime):
        nonlocal b_in, b_out
        if b_in_prime > b_in:
            raise ValueError(
                "b_in may not exceed the b_in passed into the constructor")
        return b_out

    return dp.m.make_user_measurement(
        input_domain=odometer.input_domain,
        function=function,
        input_metric=odometer.input_metric,
        output_measure=odometer.output_measure,
        privacy_map=privacy_map
    )


if __name__ == "__main__":
    import opendp.prelude as dp
    dp.enable_features("contrib", "honest-but-curious")

    # for simplicity, consider a metric space consisting of {T, F}
    domain, metric = dp.atom_domain(T=bool), dp.discrete_distance()

    odometer = make_sequential_odometer(
        domain, metric, output_measure=dp.max_divergence(T=float))

    data = True # a very simple data set
    qbl = odometer(arg=data)

    # the privacy consumption of the odometer starts at zero
    assert qbl(Map(b_in=1)) == 0

    # query the odometer queryable and release the output
    print(qbl(dp.m.make_randomized_response_bool(prob=0.75)))

    # the privacy consumption of the odometer has now increased to ln(3)
    assert qbl(Map(b_in=1)) == 1.0986122886681098



    # enforce a (b_in, b_out)-closeness continuation rule
    m_filter = make_odometer_to_filter(
        odometer=odometer, b_in=1, b_out=1.1)

    data = True # a very simple data set
    qbl = m_filter(arg=data)

    # accepted, because the total privacy spend would be less than 1.1
    qbl(dp.m.make_randomized_response_bool(prob=0.75))

    # rejected, because the total privacy spend would now exceed 1.1
    qbl(dp.m.make_randomized_response_bool(prob=0.75))




    top_level_compositor_meas = dp.c.make_sequential_composition(
        domain, metric,
        dp.max_divergence(T=float), # privacy measure
        1, [1.1, 1.1] # b_in, b_out
    )

    data = True # a very simple data set
    qbl = top_level_compositor_meas(arg=data)

    # compositor queryable accepts the privacy filter because it is (1, 1.1)-DP
    qbl_filter = qbl(m_filter)

    # filter queryable now accepts an adaptively chosen sequence of privacy losses
    qbl_filter(dp.m.make_randomized_response_bool(prob=0.6))

    # even when the top-level compositor becomes exhausted...
    qbl(dp.m.make_randomized_response_bool(prob=0.75))

    # ...the filter queryable still accepts queries, 
    # because the top-level compositor allows concurrent composition
    qbl_filter(dp.m.make_randomized_response_bool(prob=0.55))

    # rejects any query that violates the continuation rule
    qbl_filter(dp.m.make_randomized_response_bool(prob=0.75))

