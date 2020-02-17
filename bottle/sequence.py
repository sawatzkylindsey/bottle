
import pdb

from bottle.api import data
from pytils import check


# Tokens
BLANK = "<blank>"


def as_time_major(xys, y_is_sequence=True):
    check.check_iterable_of_instances(xys, data.Xy)
    maximum_length_x = max([len(xy.x) for xy in xys])
    data_x = [[] for i in range(maximum_length_x)]

    if y_is_sequence:
        maximum_length_y = max([len(xy.y) for xy in xys])
        data_y = [[] for i in range(maximum_length_y)]
    else:
        data_y = []

    for j, xy in enumerate(xys):
        for i in range(maximum_length_x):
            if i < len(xy.x):
                data_x[i] += [xy.x[i]]
            else:
                data_x[i] += [None]

        if y_is_sequence:
            for i in range(maximum_length_y):
                if i < len(xy.y):
                    data_y[i] += [xy.y[i]]
                else:
                    data_y[i] += [None]
        else:
            data_y += [xy.y]

    return data_x, data_y
