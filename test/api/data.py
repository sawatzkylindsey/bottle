
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle.api import data
from bottle.nlp import constant


def tests():
    return create_suite([LabelsTests])


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

