
import collections
import heapq
import json
import logging
import math
import numpy as np
import os
import pdb
import random

from pytils import check, stream
from bottle import util


class Xy:
    def __init__(self, x, y, name=None):
        self.x = x
        self.y = y
        self._name = name

    def __repr__(self):
        return "(x=%s, y=%s)" % (self.x, self.y)

    def name(self):
        if self._name is None:
            return str(self)
        else:
            return self._name


class Dataset:
    def __init__(self, train, validate, test):
        self.train = check.check_instance(train, Datastream)
        self.validate = check.check_instance(validate, Datastream)
        self.test = check.check_instance(test, Datastream)


class SplittingDataset:
    def __init__(self, train_split, validate_split, randomize=True):
        self.train_split = check.check_gt(train_split, 0)
        self.validate_split = check.check_gt(validate_split, 0)

        if train_split <= 0 or validate_split <= 0 or train_split + validate_split >= 1:
            raise ValueError("Invalid train/validate split combination: %f/%f" % (train_split, validate_split))

    def build(self, data_or_stream):
        data = [datum for datum in data_or_stream]
        random.shuffle(data)
        split_1 = int(len(data) * self.train_split)
        split_2 = split_1 + int(len(data) * self.validate_split)
        train_set = data[:split_1]
        validate_set = data[split_1:split_2]
        test_set = data[split_2:]
        logging.debug("Formulating t/v/t split %.2f/%.2f/%.2f: %d/%d/%d" % \
            (self.train_split, self.validate_split, 1.0 - self.train_split - self.validate_split, len(train_set), len(validate_set), len(test_set)))
        return Dataset(Datastream.from_list(train_set), \
            Datastream.from_list(validate_set), \
            Datastream.from_list(test_set))


class Datastream:
    def __init__(self, stream_fn, size=None, order_of_magnitude=None):
        self.stream_fn = check.check_function(stream_fn)
        which = check.check_exclusive({"size": size, "order_of_magnitude": order_of_magnitude})

        if which == "size":
            self.size = check.check_gte(size, 1)
            self.order_of_magnitude = util.order_of_magnitude(size)
        elif which == "order_of_magnitude":
            self.size = None
            self.order_of_magnitude = check.check_gte(order_of_magnitude, 0)
        else:
            self.size = None
            self.order_of_magnitude = None

    @classmethod
    def from_list(self, data):
        return Datastream(lambda: data, size=len(data))

    def as_randomized(self, buffer_size):
        return Datastream(lambda: stream.randomize(self.stream_fn(), buffer_size), self.size, \
            self.order_of_magnitude if self.size is None else None)

    def __repr__(self):
        details = ""

        if self.order_of_magnitude is not None:
            details = "10^%d" % self.order_of_magnitude

        if self.size is not None:
            details += " (size=%d)" % self.size

        if len(details) == 0:
            details = "..."

        return "Datastream{%s}" % details

    def _set_or_check_size(self, value):
        if self.size is None:
            self.size = value
            self.order_of_magnitude = util.order_of_magnitude(value)
        elif self.size != value:
            raise ValueError("found size %d differs from what is set %s." % (value, self.size))

    def in_batches(self, batch_size):
        s = 0
        batch = []

        for item in self.stream_fn():
            batch += [item]

            if len(batch) >= batch_size:
                s += len(batch)
                yield batch
                batch = []

        if len(batch) > 0:
            s += len(batch)
            yield batch

        self._set_or_check_size(s)

    def __iter__(self):
        self._iter_s = 0
        self._iter_stream_fn = iter(self.stream_fn())
        return self

    def __next__(self):
        try:
            item = next(self._iter_stream_fn)
            self._iter_s += 1
            return item
        except StopIteration:
            self._set_or_check_size(self._iter_s)
            self._iter_s = None
            self._iter_stream_fn = None
            raise StopIteration()

    def estimate_percent_at(self, index):
        if index < 0:
            raise ValueError("Invalid index %d." % index)

        if self.size is None and self.order_of_magnitude is None:
            return None

        if self.size is not None:
            denominator = self.size
        else:
            denominator = int(math.pow(10, self.order_of_magnitude + 1)) - 1

        # Account for the index being 0 based.
        position = float(index + 1)

        if position > denominator:
            raise ValueError("Invalid index %d for %s." % (index, self))

        return int((position / denominator) * 100)


class Field(object):
    def __init__(self):
        super(Field, self).__init__()

    def encode(self, value, handle_oov=False):
        raise NotImplementedError()

    def vector_encode(self, value, handle_oov=False):
        raise NotImplementedError()

    def decode(self, value):
        raise NotImplementedError()

    def vector_decode(self, array):
        raise NotImplementedError()


class Labels(Field):
    def __init__(self, values, special_values=None, oov_mapper=lambda word: None):
        super(Labels, self).__init__()
        check.check_set(values)
        check.check_set(special_values, noneable=True)
        self.oov_mapper = check.check_function(oov_mapper)
        self._empty = None
        self._encoding = {}
        self._decoding = {}
        labels_prefix = []
        i = 0

        if special_values is not None:
            for special in sorted(special_values):
                self._encoding[check.check_not_none(special)] = i
                self._decoding[i] = special
                labels_prefix += [special]
                i += 1

        labels = sorted([label for label in values])

        for value in labels:
            if value in self._encoding:
                raise ValueError("values and special_values must be disjoint - found both to contain '%s'." % value)

            self._encoding[check.check_not_none(value)] = i
            self._decoding[i] = value
            i += 1

        assert len(self._encoding) == len(self._decoding), "%d != %d" % (len(self._encoding), len(self._decoding))
        # Include special values in the front of the labels.
        self._labels = labels_prefix + labels
        self._encoding_copy = {k: v for k, v in self._encoding.items()}

    def __repr__(self):
        return "Labels{%s}" % self._encoding

    def __len__(self):
        return len(self._encoding)

    def encoding(self):
        return self._encoding_copy

    def labels(self):
        return self._labels

    def encode(self, value, handle_oov=False):
        try:
            return self._encoding[check.check_not_none(value)]
        except KeyError as e:
            if handle_oov:
                return self._encoding[self.oov_mapper(value)]
            else:
                raise e

    def vector_encode(self, value, handle_oov=False):
        encoding = [0] * len(self)

        if isinstance(value, dict):
            for key, probability in check.check_pdist(value).items():
                encoding[self.encode(key, handle_oov)] = probability
        else:
            encoding[self.encode(value, handle_oov)] = 1

        return np.array(encoding, dtype="float32", copy=False)

    def vector_empty(self):
        if self._empty is None:
            self._empty = np.array([0] * len(self))

        return self._empty

    def decode(self, value):
        return self._decoding[check.check_not_none(value)]

    def vector_decode(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)

        # If the array is all zeros
        if not np.any(array):
            return None

        # The index of the maximum value from the vector_encoding.
        #                  vvvvvvvvvvvvvv
        return self.decode(array.argmax())

    def sampling_vector_decode(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        # Sample 1 thing                    v
        #  from [0..N]           vvvvvvvvv
        #   with probabilties                  vvvvvvv
        index = np.random.choice(len(self), 1, p=array)[0]
        return self.decode(index)

    def vector_decode_distribution(self, array):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        return {self.decode(i): probability for i, probability in enumerate(array)}

    def vector_decode_probability(self, array, value, handle_oov=False):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        return array[self.encode(value, handle_oov)]


class VectorField(Field):
    def __init__(self, width):
        super(VectorField, self).__init__()
        self._length = width

    def __repr__(self):
        return "VectorField{%s}" % (self._length)

    def __len__(self):
        return self._length

    def encode(self, value, handle_oov=False):
        raise TypeError()

    def vector_encode(self, value, handle_oov=False):
        if len(value) != len(self):
            raise ValueError("value '%s' doesn't match vector width '%d'" % (value, self._length))

        return value

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()


class IntegerField(Field):
    def __init__(self):
        super(IntegerField, self).__init__()

    def __repr__(self):
        return "IntegerField"

    def __len__(self):
        return 1

    def encode(self, value, handle_oov=False):
        raise TypeError()

    def vector_encode(self, value, handle_oov=False):
        if not isinstance(value, int):
            raise ValueError("value '%s' isn't an integer" % value)

        return [value]

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()


class MergeLabels(Labels):
    def __init__(self, labels):
        super(MergeLabels, self).__init__(labels)
        self.labels = check.check_instance(labels, Labels)

    def __repr__(self):
        return "MergeLabels{%s}" % (self.labels)

    def __len__(self):
        return len(self.labels)

    def encoding(self):
        raise TypeError()

    def labels(self):
        raise TypeError()

    def encode(self, value, handle_oov=False):
        raise TypeError()

    def vector_encode(self, value, handle_oov=False):
        if len(value) > 0:
            # Produce the 'bitwise or' of the merge elements.
            return vector_max([self.labels.vector_encode(v, handle_oov) for v in value])
        else:
            return self.labels.vector_empty()

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()

    def sampling_vector_decode(self, array):
        raise TypeError()

    def vector_decode_distribution(self, array):
        raise TypeError()


class ConcatField(Field):
    def __init__(self, fields):
        super(ConcatField, self).__init__()
        self.fields = check.check_iterable_of_instances(fields, Field)
        self._length = sum([len(f) for f in self.fields])

    def __repr__(self):
        return "ConcatField{%s}" % ([str(f) for f in self.fields])

    def __len__(self):
        return self._length

    def encoding(self):
        raise TypeError()

    def labels(self):
        raise TypeError()

    def encode(self, value, handle_oov=False):
        raise TypeError()

    def vector_encode(self, value, handle_oov=False):
        check.check_length(value, len(self.fields))
        return np.concatenate([self.fields[i].vector_encode(v, handle_oov) for i, v in enumerate(value)])

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()

    def sampling_vector_decode(self, array):
        raise TypeError()

    def vector_decode_distribution(self, array):
        raise TypeError()


class Result:
    def __init__(self, labels, array):
        self.labels = check.check_instance(labels, Labels)
        self.array = check.check_length(array, len(self.labels))
        self._prediction = None
        self._prediction_probability = None
        self._distribution = None
        self._ranked_items = None

    def prediction(self):
        if self._prediction is None:
            self._prediction = self.labels.vector_decode(self.array)

        return self._prediction

    def prediction_probability(self):
        if self._prediction_probability is None:
            self._prediction_probability = self.array.max()

        return self._prediction_probability

    def distribution(self):
        if self._distribution is None:
            self._distribution = self.labels.vector_decode_distribution(self.array)

        return self._distribution

    def rank_of(self, value, handle_oov=False, partial_sort_k=None):
        target = self.labels.decode(self.labels.encode(value, handle_oov))

        # If the ranks haven't been determined yet, or if the partial sort specified extends beyond what has been already determined.
        if self._ranked_items is None or (partial_sort_k is not None and len(self._ranked_items) < partial_sort_k):
            distribution_items = [item for item in self.distribution().items()]

            if partial_sort_k is not None:
                # Partial sort mechanism 'nlargest'.
                largest = heapq.nlargest(partial_sort_k, distribution_items, key=lambda item: item[1])
                self._ranked_items = {item[0]: rank for rank, item in enumerate(sorted(largest, key=lambda item: item[1], reverse=True))}
            else:
                self._ranked_items = {item[0]: rank for rank, item in enumerate(sorted(distribution_items, key=lambda item: item[1], reverse=True))}

        if target in self._ranked_items:
            return self._ranked_items[target]
        else:
            # The target doesn't fit within the partial sort.
            # Give it the last rank.
            return len(self.labels) - 1

    def __repr__(self):
        return "(.., prediction=%s)" % (self.prediction())

