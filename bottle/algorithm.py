
import math
import numpy as np
import pdb


def vector_max(vectors):
    length = None

    for ook in vectors:
        if length is None:
            length = len(ook)
        else:
            assert len(ook) == length, "%d != %d" % (len(ook), length)

    def _max(bits):
        out = bits[0]

        for bit in bits:
            if bit > out:
                out = bit

        return out

    out = np.array([_max([ook[i] for ook in vectors]) for i in range(length)])
    assert len(out) == length, "%d != %d" % (len(out), length)
    return out


def softmax(distribution):
    total = 0.0
    output = {}

    for k, v in distribution.items():
        value = math.exp(v)
        output[k] = value
        total += value

    return {k: v / total for k, v in output.items()}


def regmax(distribution):
    total = 0.0
    output = {}

    for k, v in distribution.items():
        value = v
        output[k] = value
        total += value

    return {k: v / total for k, v in output.items()}


# Return the index of the target, or where it should be inserted, based of an input array sorted descending.
def binary_search(descending_array, target, accessor=lambda item: item):
    if len(descending_array) == 0:
        return 0

    lower = 0
    upper = len(descending_array) - 1
    found = upper + 1 if accessor(descending_array[upper]) > target else None

    if found is None and accessor(descending_array[lower]) < target:
        found = 0

    while found is None:
        current = int((upper + lower) / 2.0)
        observation = accessor(descending_array[current])

        if observation == target:
            found = current + 1
        elif observation < target:
            upper = current
        else:
            lower = current

        if lower + 1 == upper:
            found = upper

    return found

