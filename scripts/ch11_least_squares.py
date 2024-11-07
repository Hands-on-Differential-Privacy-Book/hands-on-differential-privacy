import pandas as pd
import numpy as np

data = pd.read_csv('sequestered_data.csv')


def query_interface(predicates, target):
    """Count the number of smokers that satisfy each predicate.
    Resembles a public query interface on a sequestered data set.

    :param predicates: a list of predicates on the public variables
    :param target: column to filter against
    :returns a 1-d np.ndarray of exact answers to the subset sum queries"""

    # 1. data curator checks predicates
    # 2. data curator executes and returns queries:
    query_matrix = np.stack([pred(data) for pred in predicates], axis=1)
    return data[target].values @ query_matrix


query_interface([
    lambda data: data['sex'] == 1,            # "is-female" predicate
    lambda data: data['maritalStatus'] == 1,  # "is-married" predicate
], target="smoker")


# TODO: define pub
pub = None


def make_random_predicate():
    """Returns a (pseudo)random predicate function by
       hashing public identifiers."""
    prime = 691
    desc = np.random.randint(prime, size=len(pub))
    # this predicate maps data into a 1-d ndarray of booleans
    #   (where `@` is the dot product and `%` modulus)
    return lambda data: ((data[pub].values @ desc) % prime % 2).astype(bool)

# Example usage
random_predicate = make_random_predicate()
num_smokers_that_matched_random_predicate = query_interface([random_predicate],
                                                            "smoker")

# The boolean mask from applying the example predicate to the data:
random_predicate_mask = random_predicate(data)


def reconstruction_attack(data_pub, predicates, answers):
    """Reconstructs a target column based on the `answers` to queries
       about `data`.

    :param data_pub: data of length n consisting of public identifiers
    :param predicates: a list of k predicate functions
    :param answers: a list of k answers to a query on data filtered by the
                    k predicates
    :return 1-dimensional boolean ndarray"""
    masks = np.stack([pred(data_pub) for pred in predicates])
    return np.linalg.lstsq(masks, answers, rcond=None)[0] > 0.5


predicates = [make_random_predicate() for _ in range(2 * len(data))]
exact_answers = query_interface(predicates, "smoker")

# generate example predicates and compute example query answers
reconstructed_target = reconstruction_attack(
    data_pub=data[pub], predicates=predicates, answers=exact_answers)

target = 'smoker'
# complete reconstruction of the target column
assert np.array_equal(reconstructed_target, data[target])
