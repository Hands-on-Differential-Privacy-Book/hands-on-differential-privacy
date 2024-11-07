import opendp.prelude as dp

# call a constructor to produce a transformation
stable_trans = dp.t.make_sum(
    dp.vector_domain(dp.atom_domain(bounds=(0, 1))),
    dp.symmetric_distance())

# call a constructor to produce a measurement
private_mech = dp.m.make_laplace(
    stable_trans.output_domain, 
    stable_trans.output_metric, 
    scale=1.0)

new_mech = dp.c.make_chain_mt(private_mech, stable_trans)

# investigate the privacy relation
symmetric_distance = 1
epsilon = 1.0
assert new_mech.map(symmetric_distance) <= epsilon

# invoke the chained measurement's function
mock_data = [0, 0, 1, 1, 0, 1, 1, 1]
release = new_mech(mock_data)