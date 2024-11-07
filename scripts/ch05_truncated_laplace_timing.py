import opendp.prelude as dp
from ch05_truncated_laplace import make_truncated_laplace

def make_private_timing(meas, t, scale, delay):
    dp.assert_features("contrib")
    import time
    if meas.output_measure.type.origin != \
            "FixedSmoothedMaxDivergence":
        raise ValueError("measurement must give "
                         "privacy guarantees wrt (ε, δ)")
    
    tlap = make_truncated_laplace(scale, delay)

    def f_add_timing_delay(arg):
        time.sleep(tlap(delay)) # random delay privatizes elapsed time
        return meas(arg)

    def privacy_map(b_in):
        (eps_1, del_1), (eps_2, del_2) = meas.map(b_in), tlap.map(t)
        return eps_1 + eps_2, del_1 + del_2
    
    return dp.m.make_user_measurement(
        input_domain=meas.input_domain,
        input_metric=meas.input_metric,
        output_measure=meas.output_measure,
        function=f_add_timing_delay,
        privacy_map=privacy_map)
