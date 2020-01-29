
import collections
import heapq
import json
import logging
import math
import numpy as np
import os
import pdb

from pytils import check
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


class Datastream:
    def __init__(self, stream_fn, size=None, order_of_magnitude=None):
        self.stream_fn = stream_fn
        which = check.check_exclusive({"size": size, "order_of_magnitude": order_of_magnitude})

        if which == "size":
            self.size = check.check_gte(size, 1)
            self.order_of_magnitude = util.order_of_magnitude(size)
        else:
            self.size = None
            self.order_of_magnitude = check.check_gte(order_of_magnitude, 0)

    def _set_or_check_size(self, value):
        if self.size is None:
            self.size = value
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


class Field(object):
    def __init__(self):
        super(Field, self).__init__()

    def encode(self, value, handle_unknown=False):
        raise NotImplementedError()

    def vector_encode(self, value, handle_unknown=False):
        raise NotImplementedError()

    def decode(self, value):
        raise NotImplementedError()

    def vector_decode(self, array):
        raise NotImplementedError()


class Labels(Field):
    def __init__(self, values, unknown=None):
        super(Labels, self).__init__()
        check.check_set(values)
        self.unknown = unknown
        self._empty = None
        self._encoding = {}
        self._decoding = {}
        labels_prefix = []

        if unknown is not None:
            self._encoding[unknown] = 0
            self._decoding[0] = self.unknown
            labels_prefix = [unknown]

        i = len(self._encoding)
        labels = sorted([label for label in values])

        for value in labels:
            if unknown is None or value != unknown:
                self._encoding[check.check_not_none(value)] = i
                self._decoding[i] = value
                i += 1

        assert len(self._encoding) == len(self._decoding), "%d != %d" % (len(self._encoding), len(self._decoding))
        # Include unknown in the correct position if it's being represented in the labels.
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

    def encode(self, value, handle_unknown=False):
        try:
            return self._encoding[check.check_not_none(value)]
        except KeyError as e:
            if handle_unknown:
                return self._encoding[self.unknown]
            else:
                raise e

    def vector_encode(self, value, handle_unknown=False):
        encoding = [0] * len(self)

        if isinstance(value, dict):
            for key, probability in check.check_pdist(value).items():
                encoding[self.encode(key, handle_unknown)] = probability
        else:
            encoding[self.encode(value, handle_unknown)] = 1

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

    def vector_decode_probability(self, array, value, handle_unknown=False):
        assert len(array) == len(self), "%d != %d" % (len(array), len(self))
        check.check_pdist(array)
        return array[self.encode(value, handle_unknown)]


class VectorField(Field):
    def __init__(self, width):
        super(VectorField, self).__init__()
        self._length = width

    def __repr__(self):
        return "VectorField{%s}" % (self._length)

    def __len__(self):
        return self._length

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
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

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
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

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
        if len(value) > 0:
            # Produce the 'bitwise or' of the merge elements.
            return vector_max([self.labels.vector_encode(v, handle_unknown) for v in value])
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

    def encode(self, value, handle_unknown=False):
        raise TypeError()

    def vector_encode(self, value, handle_unknown=False):
        check.check_length(value, len(self.fields))
        return np.concatenate([self.fields[i].vector_encode(v, handle_unknown) for i, v in enumerate(value)])

    def decode(self, value):
        raise TypeError()

    def vector_decode(self, array):
        raise TypeError()

    def sampling_vector_decode(self, array):
        raise TypeError()

    def vector_decode_distribution(self, array):
        raise TypeError()

