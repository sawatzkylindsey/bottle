
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle.api import data
from bottle.nlp import constant


def tests():
    return create_suite([LabelsTests, DatastreamTests])


class LabelsTests(TestCase):
    def test_labels(self):
        values = ["word", "xyz"]
        labels = data.Labels(set(values))
        self.assertEqual(labels.labels(), sorted(values))
        self.assertEqual(sorted([
            labels.encode("word"),
            labels.encode("xyz"),
        ]), [0, 1])  # Don't care about which labels are encoded as which indexes.

        with self.assertRaises(KeyError):
            labels.encode("not-a-word")

        with self.assertRaises(KeyError):
            labels.encode("not-a-word", handle_oov=True)

    def test_labels_special(self):
        values = ["word", "xyz"]
        special_values = ["qwerty", "abc"]
        labels = data.Labels(set(values), set(special_values))
        self.assertEqual(labels.labels(), sorted(special_values) + sorted(values))
        self.assertEqual(sorted([
            labels.encode("word"),
            labels.encode("xyz"),
            labels.encode("qwerty"),
            labels.encode("abc"),
        ]), [0, 1, 2, 3])  # Don't care about which labels are encoded as which indexes.

    def test_labels_oov(self):
        values = ["word", "xyz"]
        labels = data.Labels(set(values), set([constant.UNKNOWN]), oov_mapper=lambda word: constant.UNKNOWN)

        with self.assertRaises(KeyError):
            labels.encode("not-a-word")

        self.assertEqual(labels.encode("not-a-word", handle_oov=True), labels.encode(constant.UNKNOWN))


class DatastreamTests(TestCase):
    def test_datastream(self):
        datastream = data.Datastream(lambda: range(3), size=3)
        self.assertEqual(datastream.size, 3)
        self.assertEqual(datastream.order_of_magnitude, 0)
        items = [item for item in datastream]

        self.assertEqual(items, [i for i in range(3)])
        self.assertEqual(datastream.size, 3)
        self.assertEqual(datastream.order_of_magnitude, 0)

        # Check that it can be re-streamed
        items = [item for item in datastream]
        self.assertEqual(items, [i for i in range(3)])

    def test_batches(self):
        datastream = data.Datastream(lambda: range(3), size=3)
        items = []

        for batch in datastream.in_batches(2):
            if len(batch) != 1 and len(batch) != 2:
                raise AssertionError("incorrect batching")

            for item in batch:
                items += [item]

        self.assertEqual(items, [i for i in range(3)])

    def test_order_of_magnitude(self):
        datastream = data.Datastream(lambda: range(11), order_of_magnitude=1)
        self.assertEqual(datastream.size, None)
        self.assertEqual(datastream.order_of_magnitude, 1)
        items = []

        for batch in datastream.in_batches(2):
            if len(batch) != 1 and len(batch) != 2:
                raise AssertionError("incorrect batching")

            for item in batch:
                items += [item]

        self.assertEqual(items, [i for i in range(11)])
        self.assertEqual(datastream.size, 11)
        self.assertEqual(datastream.order_of_magnitude, 1)

    def test_estimate_percent_at(self):
        datastream = data.Datastream.from_list([i for i in range(100)])

        for i, item in enumerate(datastream):
            percent_complete = datastream.estimate_percent_at(i)
            # Simple +/- mechanism.
            value = max(percent_complete - i - 1, 0)
            self.assertEqual(value, 0)

        # Check that it can be re-streamed
        self.assertEqual([item for item in datastream], [i for i in range(100)])

    def test_estimate_percent_at_order_of_magnitude(self):
        items = [i for i in range(100)]
        datastream = data.Datastream(lambda: items, order_of_magnitude=2)

        for i, item in enumerate(datastream):
            percent_complete = datastream.estimate_percent_at(i)
            # Simple +/- mechanism.
            value = max(percent_complete - i - 1, 0)
            self.assertEqual(value, 0)

        # Check that it can be re-streamed
        self.assertEqual([item for item in datastream], [i for i in range(100)])

