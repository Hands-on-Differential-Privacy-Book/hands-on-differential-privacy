import opendp.prelude as dp


def make_information_gain(attributes):
    def function(arg):
        # TODO: compute the information gain from each attribute
        pass

    def stability_map(b_in):
        # TODO: derive the sensitivity of the information gain
        pass

    dp.t.make_user_transformation(
        input_domain=array_domain(),
        input_metric=dp.symmetric_distance(),
        output_domain=dp.vector_domain(dp.atom_domain(T=float)),
        output_metric=dp.linf_distance(T=float),
        function=function,
        stability_map=stability_map)
