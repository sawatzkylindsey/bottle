
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

    def __iter__(self):
        return iter(self.queue)

    def __len__(self):
        return len(self.queue)

    def __getitem__(self, index):
        return self.queue[index]

    def __repr__(self):
        return str([str(v) for v in self.queue])

