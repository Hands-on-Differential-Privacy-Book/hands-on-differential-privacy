from math import sqrt
import opendp.prelude as dp


def plrv_gaussian_composition(scales):
    def privacy_map(b_in):
        eta_g = sum(b_in**2 / (2 * s_i**2) for s_i in scales)
        scale_g = sqrt(b_in**2 / (2 * eta_g))
        
        space = dp.atom_domain(T=float), dp.absolute_distance(T=float)
        return dp.m.make_base_gaussian(*space, scale_g).map(b_in)

    return privacy_map
