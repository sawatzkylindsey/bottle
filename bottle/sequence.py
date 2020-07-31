
import math
import pdb

from pytils import check

from bottle.api import data
from bottle.nlp import constant, processing


# Tokens
BLANK = "<blank>"

ZERO_PROBABILITY = 1e-100


class WordLabels(data.Labels):
    def __init__(self, dictionary):
        super().__init__(dictionary, set([BLANK, constant.UNKNOWN, constant.NUMBER]), lambda word: constant.NUMBER if processing.is_number(word) else constant.UNKNOWN)


def perplexity(probabilities):
    check.check_not_empty(probabilities)
    total_log_probability = 0.0

    for probability in probabilities:
        if probability < 0.0 or probability > 1.0:
            raise ValueError("Invalid probability [0, 1]: %f." % probability)

        total_log_probability += math.log2(ZERO_PROBABILITY if probability == 0 else probability)

    return math.pow(2.0, -total_log_probability / len(probabilities))


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


class DebugStats:
    def __init__(self, sequence, predicted, expected):
        self.sequence = check.check_not_empty(sequence)
        self.predicted = check.check_instance(predicted, SequenceStats)
        self.expected = check.check_instance(expected, SequenceStats)
        check.check_length(self.predicted.values, len(self.sequence))
        check.check_length(self.expected.values, len(self.sequence))
        self.perplexity = perplexity(self.expected.probabilities)

    def as_formatted(self):
        float_points = 4
        string_lengths = []

        for timestep in range(len(self.sequence)):
            maximum = float_points + 2

            if len(self.sequence[timestep]) > maximum:
                maximum = len(self.sequence[timestep])

            if len(self.predicted.values[timestep]) > maximum:
                maximum = len(self.predicted.values[timestep])

            if len(self.expected.values[timestep]) > maximum:
                maximum = len(self.expected.values[timestep])

            string_lengths += [maximum]

        debug_template = "  ".join(["{:%d.%ds}" % (l, l) for l in string_lengths])
        float_template = "{:.%df}" % float_points
        int_template = "{:d}"
        sequence_str = debug_template.format(*(self.sequence))
        predicted_str = debug_template.format(*self.predicted.values)
        predicted_probability_str = debug_template.format(*[float_template.format(p) for p in self.predicted.probabilities])
        expected_str = debug_template.format(*self.expected.values)
        expected_probability_str = debug_template.format(*[float_template.format(p) for p in self.expected.probabilities])
        expected_rank_str = None

        if self.expected.ranks is not None:
            expected_rank_str = debug_template.format(*[int_template.format(i) for i in self.expected.ranks])

        details =  ["Calculated perplexity: %f" % self.perplexity]
        details += ["  Sequence: %s" % sequence_str.strip()]
        details += [" Predicted: %s" % predicted_str.strip()]
        details += ["          : %s" % predicted_probability_str.strip()]
        details += ["  Expected: %s" % expected_str.strip()]
        details += ["          : %s" % expected_probability_str.strip()]

        if expected_rank_str is not None:
            details += ["      rank: %s" % expected_rank_str.strip()]

        return "\n".join(details)


class SequenceStats:
    def __init__(self, values, probabilities, ranks=None):
        self.values = check.check_iterable(values)
        self.probabilities = check.check_length(check.check_iterable(probabilities), len(self.values))
        self.ranks = check.check_length(check.check_iterable(ranks, noneable=True), len(self.values), noneable=True)

