import opendp.prelude as dp
import random

def make_resize(target_size, constant):
    def f_resize(x):
        if len(x) > target_size:
            random.shuffle(x) # data should be considered unordered
            return x[:target_size]
        if len(x) < target_size:
            return x + [constant] * (target_size - len(x))
        return x
    
    atom_domain = dp.atom_domain(T=type(constant))

    return dp.t.make_user_transformation(
        input_domain=dp.vector_domain(atom_domain), 
        input_metric=dp.symmetric_distance(), 
        output_domain=dp.vector_domain(atom_domain, size=target_size),
        output_metric=dp.symmetric_distance(),
        function=f_resize,
        stability_map=lambda b_in: 2 * b_in
    )
