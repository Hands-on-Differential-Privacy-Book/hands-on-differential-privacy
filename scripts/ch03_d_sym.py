from collections import Counter

def d_Sym(x, x_p):
    """symmetric distance between x and x'"""
    # NOT this, as sets are not multisets. Loses multiplicity:
    # return len(set(x).symmetric_difference(set(x_p)))
    u_counter, v_counter = Counter(x), Counter(x_p)
    # indirectly compute symmetric difference via union of asymmetric differences
    return sum(((u_counter - v_counter) + (v_counter - u_counter)).values())
