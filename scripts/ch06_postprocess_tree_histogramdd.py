import numpy as np


def _axes_to_sum(child, parent):
    # Find the indexes of axes that should be summed in `child` to get `parent`
    return tuple(child.index(i) for i in child if i not in parent)


def _branching_factor(category_lengths, axes_to_sum):
    # branching factor between parent and child 
    #    is the product of lengths of collapsed axes
    return np.prod(category_lengths[np.array(axes_to_sum)])


def postprocess_tree_histogramdd(hists, std):
    """Make a set of noisy hypercubes of successive summations
    consistent with each other.
    See 4.1: https://arxiv.org/pdf/0904.0942.pdf

    :param hists: A dict of {axes: hist}, 
        where each hist is a `len(axes)`-dimensional counts array.
    :returns the leaf layer histogram
    """
    # sort the keys by number of axes
    hists = dict(sorted(hists.items(), key=lambda p: len(p[0])))
    # ensure all hists are float
    hists = {k: v.astype(float) for k, v in hists.items()}

    axes = list(hists)

    # find shape of each axis. Last histogram holds all axis lengths
    category_lengths = np.array(hists[axes[-1]].shape)

    # variance of postprocessed current layer. Starting at leaves,
    # which are not postprocessed
    var = std[axes[-1]] ** 2
    # ...

    # ...
    # bottom-up scan to compute z
    for parent, child in zip(axes[::-1][1:], axes[::-1][:-1]):
        axes_to_sum = _axes_to_sum(child=child, parent=parent)
        b = _branching_factor(category_lengths, axes_to_sum)

        # derive overall variance of parent after weighted averaging
        var = 1 / (1 / std[parent]**2 + 1 / (b * var))

        # weight parent contribution based on its proportion of inverse variance
        alpha = var / std[parent]**2

        # hists[parent] has not been overriden because traversal order
        # is bottom to top
        term1 = alpha * hists[parent]

        # hists[child] has been overwritten by previous loop
        term2 = (1 - alpha) * hists[child].sum(axis=axes_to_sum)

        hists[parent] = term1 + term2
    # ...

    # ...
    h_b = {a: h.copy() for a, h in hists.items()}

    # top down scan to compute h
    for parent, child in zip(axes[:-1], axes[1:]):
        axes_to_sum = _axes_to_sum(child=child, parent=parent)
        b = _branching_factor(category_lengths, axes_to_sum)

        correction = (h_b[parent] - hists[child].sum(axis=axes_to_sum)) / b
        h_b[child] += np.expand_dims(correction, axes_to_sum)

    # entire tree is consistent, so only the bottom layer is needed
    return h_b[axes[-1]]