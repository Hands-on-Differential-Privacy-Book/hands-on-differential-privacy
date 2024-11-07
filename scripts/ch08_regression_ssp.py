import opendp.prelude as dp
import numpy as np

def make_insert_ones():
    dp.assert_features("contrib")
    space = dp.np_array2_domain(T=float), dp.symmetric_distance()
    return dp.t.make_user_transformation(
        *space, *space,
        function=lambda X: np.insert(X, 0, 1., axis=1),
        stability_map=lambda b_in: b_in)   


def make_xTx(norm):
    dp.assert_features("contrib", "floating-point")
    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(origin=0., norm=norm, ord=2),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.l2_distance(T=float),
        function=lambda X: (X.T @ X).ravel(),
        stability_map=lambda b_in: b_in * norm**2)


def beta_postprocessor(ZTZ):
    ZTZ = np.array(ZTZ).reshape([int(np.sqrt(len(ZTZ)))] * 2) # make it square
    ZTZ = (ZTZ + ZTZ.T) / 2 # symmetrizes, and halves variance off-the-diagonal
    XTX, XTy = ZTZ[:-1, :-1], ZTZ[:-1, -1] # extract point estimates
    return np.linalg.pinv(XTX) @ XTy # plug into MVUE estimator for β


def make_private_beta(norm, scale):
    return (
        make_insert_ones() >>            # 1-stable
        dp.t.then_np_clamp(norm=norm) >> # 1-stable from OpenDP Library
        make_xTx(norm) >>                # b^2-stable aggregator
        dp.m.then_gaussian(scale) >>     # mechanism from OpenDP Library
        beta_postprocessor)


dp.enable_features("honest-but-curious", "contrib", "floating-point")
meas = make_private_beta(1., scale=10.)
data = np.random.normal(size=(10_000, 5))
print(meas(data))
