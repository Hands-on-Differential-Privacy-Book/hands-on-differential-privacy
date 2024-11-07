import opendp.prelude as dp


from ch05_rdp_to_fixed_approx_dp import renyi_divergence


def make_zCDP_to_RDP(meas):
    """Convert a zCDP mechanism to an RDP mechanism"""
    dp.assert_features("contrib")
    assert meas.output_measure == dp.zero_concentrated_divergence(T=float)

    def privacy_map(b_in):
        # ρ-zCDP implies (α, ρ * α)-RDP
        def rdp_curve(alpha):
            assert alpha > 1, "RDP order (alpha) must be greater than one"
            return meas.map(b_in) * alpha
        return rdp_curve
    
    return dp.m.make_user_measurement(
        meas.input_domain, meas.input_metric, renyi_divergence(),
        function=meas.function, privacy_map=privacy_map)

if __name__ == "__main__":
    import opendp.prelude as dp
    space = dp.atom_domain(T=float), dp.absolute_distance(T=float)
    meas_pureDP = space >> dp.m.then_base_laplace(scale=10.)

    # convert the output measure to `FixedSmoothedMaxDivergence`
    meas_fixed_approxDP = dp.c.make_pureDP_to_fixed_approxDP(meas_pureDP)

    # FixedSmoothedMaxDivergence distances are (ε, δ) tuples
    meas_fixed_approxDP.map(1.) # -> (0.1, 0.0)



    meas_zCDP = space >> dp.m.then_base_gaussian(scale=0.5)

    # convert the output measure to approx-DP
    meas_approxDP = dp.c.make_zCDP_to_approxDP(meas_zCDP)

    # this distance in approx-DP is represented with an ε(δ) curve
    εδ_curve = meas_approxDP.map(1.)
