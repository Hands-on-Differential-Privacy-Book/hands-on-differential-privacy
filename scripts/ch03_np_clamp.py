import numpy as np
import opendp.prelude as dp


def make_np_clamp(norm, p, origin=None):
    dp.assert_features("contrib", "floating-point")
    assert norm >= 0., "norm must not be negative"

    # assume the origin is at zero if not specified
    origin = 0.0 if origin is None else origin

    def clamp_row_norms(data):
        data = data.copy()
        # shift the data around zero
        data -= origin

        # compute the p-norm of each row
        row_norms = np.linalg.norm(data, ord=p, axis=1, keepdims=True)
        # scale each row down to have norm at most 1
        data /= np.maximum(row_norms / norm, 1)
        
        # shift the normed data around zero back to `origin`
        data += origin
        return data

    return dp.t.make_user_transformation(
        input_domain=dp.np_array2_domain(T=float), # input data is unconstrained
        input_metric=dp.symmetric_distance(),
        output_domain=dp.np_array2_domain(norm=norm, p=p, origin=origin),
        output_metric=dp.symmetric_distance(),
        function=clamp_row_norms, 
        stability_map=lambda b_in: b_in) # norm clamping is 1-stable row-by-row
