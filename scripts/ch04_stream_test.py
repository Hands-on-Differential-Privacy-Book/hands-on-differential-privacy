
from ch04_stream import make_above_threshold, make_query_stream
import numpy as np
import opendp.prelude as dp
dp.enable_features("honest-but-curious", "contrib", "floating-point")


def test_above_threshold():
    meas = make_above_threshold(threshold=4, scale=1.)
    # since the measurement privatizes a stream, initialize with a dummy stream
    qbl = meas(lambda x: x)

    qbl(2) # very likely to emit False
    qbl(3) # likely to emit False
    qbl(6) # likely to emit True for the first time
    qbl(4) # expected to throw an assertion error


def test_make_query_stream():
    space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
    meas = make_query_stream(*space, b_in=1, b_out=1)
    qbl = meas(np.random.randint(100, size=100))

    print(qbl(dp.t.make_count_distinct(*space)))
    print(qbl(dp.t.make_count(*space)))


def test_query_above_threshold():
    # construct the private mechanism
    space = dp.vector_domain(dp.atom_domain(T=int)), dp.symmetric_distance()
    meas = make_query_stream(*space, b_in=1, b_out=1) >> \
           make_above_threshold(threshold=4, scale=1.)
    
    # begin analysis of sensitive data
    qbl = meas([1, 2, 2, 2, 3])
    print(qbl(dp.t.make_count_distinct(*space))) # likely to emit False
    print(qbl(dp.t.make_count(*space))) # likely to emit True

