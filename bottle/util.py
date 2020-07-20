
import collections
import math


def order_of_magnitude(value):
    return int(math.log10(value))


def change_to_str(previous, current):
    if previous is None:
        return "/"
    elif previous == current:
        return "-"
    elif previous > current:
        return "▼"
    else:
        return "▲"


def average_slope(values):
    if len(values) <= 1:
        raise ValueError("Cannot calculate slope from only %d points." % len(values))

    sliding_slopes = [values[i + 1] - values[i] for i in range(len(values) - 1)]
    return sum(sliding_slopes) / float(len(sliding_slopes))


class Window:
    def __init__(self, size):
        self.size = size
        self.queue = collections.deque([])

    def append(self, value):
        self.queue.append(value)

        if len(self.queue) > self.size:
            self.queue.popleft()

    def append_all(self, values):
        for value in values:
            self.append(value)

    def is_full(self):
        return self.size == len(self.queue)

    def __iter__(self):
        return iter(self.queue)

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, index):
        return self.queue[index]

    def __repr__(self):
        return str([str(v) for v in self.queue])

