from typing import Iterable
import random

def reservoir_sample(stream: Iterable, k: int):
    """sample `k` items uniformly at random from `stream`"""
    # take up to the first k items from the stream
    reservoir = [v for v, _ in zip(stream, range(k))]

    for i, element in enumerate(stream, start=k):
        # uniformly sample from [0, item_position]
        j = random.randrange(i + 1)

        # if the sampled index `j` is in the reservoir...
        if j < k:
            # ...then replace the element in the reservoir
            reservoir[j] = element

    return reservoir


import numpy as np


whitelist = {*range(10), 71, 72, 99, 100, 107, 108, 643, 644, 825, 826, 915, 916}
def instrumented_reservoir_sample(stream: Iterable, k: int):
    reservoir = np.array([v for v, _ in zip(stream, range(k))])

    print("| i| j|reservoir")
    print(f"|  |  |[ 0  1  2  3  4  5  6  7  8  9]")
    for i, element in enumerate(stream):
        # uniformly sample from [0, item_position]
        j = random.randrange(i + k + 1)

        # if the sampled index `j` is in the reservoir...
        if j < k:
            # ...then replace the element in the reservoir
            reservoir[j] = element

        if i in whitelist:
            print(f"|{i + 1:2}|{j:2}|{reservoir}")


instrumented_reservoir_sample(iter(range(1000)), k=10)
