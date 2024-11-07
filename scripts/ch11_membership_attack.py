import numpy as np


def membership_attack(individual, answers, reference_samples, alpha=.05):
    """Perform membership attack using dwork et al. test statistic.
    See figure 1 in
    https://privacytools.seas.harvard.edu/files/privacytools/files/robust.pdf

    :param individual: y, a boolean vector of shape (d,)
    :param answers: q, a float vector with elements in [0, 1] of shape (d,)
    :param reference_samples: z, a boolean vector of length (1, d)
    :param alpha: statistical significance
    :return: True if alice is in data with (1-alpha)100% confidence."""
    individual = individual * 2 - 1
    answers = answers * 2 - 1
    reference_samples = reference_samples * 2 - 1

    alice_similarity = np.dot(individual, answers) # cosine similarity
    reference_similarity = np.dot(reference_samples[0], answers)
    statistic = alice_similarity - reference_similarity

    d = len(individual)
    tau = np.sqrt(8 * d * np.log(1 / alpha))
    return statistic > tau