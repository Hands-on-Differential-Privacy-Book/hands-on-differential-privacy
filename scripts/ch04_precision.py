import opendp.prelude as dp
dp.enable_features("contrib", "honest-but-curious")

def make_tp_fp_tn_fn(expect):
    def label(e_i, p_i):
        return ("T" if e_i == p_i else "F") + ("P" if p_i else "N")
    
    n = len(expect)
    return dp.t.make_user_transformation(    
        input_domain=dp.vector_domain(dp.atom_domain(T=bool), size=n),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=str), size=n),
        output_metric=dp.symmetric_distance(),
        function=lambda actual: [label(e, p)
                                 for e, p in zip(expect, actual)],
        stability_map=lambda b_in: b_in  # 1-stable
    )

def precision_postprocess(tp_fp_tn_fn):
    tp, fp, *_ = tp_fp_tn_fn
    return tp / (tp + fp)

import numpy as np
expect = np.random.choice(a=[False, True], size=1000)
actual = np.random.choice(a=[False, True], size=1000)

categories = ["TP", "FP", "TN", "FN"]
meas = (
    make_tp_fp_tn_fn(expect) >> # 1-stable labeling
    dp.t.then_count_by_categories(categories, null_category=False) >>
    dp.m.then_laplace(2.) >>
    precision_postprocess
)
print(meas(actual))     # ~> 0.4763 precision
print(meas.map(2)) # -> 1.0 epsilon
