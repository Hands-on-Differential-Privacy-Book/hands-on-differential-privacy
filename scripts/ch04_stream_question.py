from ch04_stream import make_above_threshold, queryable_domain
import opendp.prelude as dp
TODO = "fix usages of this variable"

def make_sum_stream(bounds):
    dp.assert_features("contrib")
    T, L, U = type(bounds[0]), *bounds
    def f_sum_stream(stream):
        transition = lambda b_in: TODO # implement the new stream
        return dp.new_queryable(transition, Q=dp.Vec[T], A=T)
    
    stream_output_domain = dp.vector_domain(dp.atom_domain(bounds=bounds))
    return dp.t.make_user_transformation(
        input_domain=queryable_domain(stream_output_domain),
        # technically l-infty of sym dists
        input_metric=dp.symmetric_distance(), 
        output_domain=queryable_domain(dp.atom_domain(T=float)),
        output_metric=dp.linf_distance(T=T, monotonic=True),
        function=f_sum_stream,
        stability_map=TODO # what is the stability of the new stream?
    )

meas = TODO # construct the complete measurement
assert meas.map(1) <= 1, "ε should be at most 1"
qbl = meas(lambda x: x) # `meas` will privatize the identity stream

#        Day 1,        Day 2,      DAY 3 (BIG), Day 4 (likely never seen)
sales = [[9.27, 9.32], [1.34] * 3, [8.92] * 30, [12.73, 8.34, 7.32], ...]
# scan the sales data stream until the mechanism detects high sales
print("high sales on day:", 
      next(i + 1 for i, s in enumerate(sales) if qbl(s)))