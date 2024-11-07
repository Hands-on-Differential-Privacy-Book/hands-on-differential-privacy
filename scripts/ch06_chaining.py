import opendp.prelude as dp


def make_chain_pm(f_2, f_1):
    """Construct a new postprocessed measurement representing f_c(x) = f_2(f_1(x))"""
    # check that f_1 is a measurement
    assert isinstance(f_1, dp.Measurement)

    # transformation's output space must conform with measurement's input space
    assert f_2.input_domain == f_1.output_domain
    assert f_2.input_metric == f_1.output_metric

    # construct a new measurement representing the functional composition
    return dp.m.make_user_measurement(
        input_domain=f_1.input_domain,
        input_metric=f_1.input_metric,
        output_measure=f_2.output_measure,
        function=lambda x: f_2(f_1(x)),
        privacy_map=lambda b_in: f_2.map(f_1.map(b_in)))


def make_chain_mt(f_2, f_1):
    """Construct a new measurement representing f_c(x) = f_2(f_1(x))"""
    # check that f_1 is a transformation and f_2 is a measurement
    assert isinstance(f_2, dp.Measurement) and isinstance(f_1, dp.Transformation)

    # transformation's output space must conform with measurement's input space
    assert f_2.input_domain == f_1.output_domain
    assert f_2.input_metric == f_1.output_metric

    # construct a new measurement representing the functional composition
    return dp.m.make_user_measurement(
        input_domain=f_1.input_domain,
        input_metric=f_1.input_metric,
        output_measure=f_2.output_measure,
        function=lambda x: f_2(f_1(x)),
        privacy_map=lambda b_in: f_2.map(f_1.map(b_in)))


def make_chain_tt(f_2, f_1):
    """Construct a new transformation representing f_c(x) = f_2(f_1(x))"""
    # check that both f_1 and f_2 are transformations
    assert isinstance(f_2, dp.Transformation) and isinstance(f_1, dp.Transformation)

    # transformation's output space must conform with transfomration's input space
    assert f_2.input_domain == f_1.output_domain
    assert f_2.input_metric == f_1.output_metric

    # construct a new transformation representing the functional composition
    return dp.t.make_user_transformation(
        input_domain=f_1.input_domain,
        input_metric=f_1.input_metric,
        output_domain=f_2.output_domain,
        output_metric=f_2.output_metric,
        function=lambda x: f_2(f_1(x)),
        stability_map=lambda b_in: f_2.map(f_1.map(b_in)))
