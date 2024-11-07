import opendp.prelude as dp
dp.enable_features("honest-but-curious")

space = dp.vector_domain(dp.atom_domain(T=float, size=10)), 
        dp.symmetric_distance()
meas = (
    space >>
    dp.t.then_clamp((0., 10.)) >>
    dp.t.then_mean() >>
    dp.m.then_laplace(scale=0.5)
)

amplified = dp.c.make_population_amplification(meas, population_size=100)

# Where we once had a privacy utilization of ~2 epsilon...
assert meas.map(2) <= 2. + 1e-6

# ...we now have a privacy utilization of ~.4941 epsilon.
assert amplified.map(2) <= .4941
