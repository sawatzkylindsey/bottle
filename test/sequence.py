
import math
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle import sequence


def tests():
    return create_suite([PerplexityTests])


class PerplexityTests(TestCase):
    def test_perplexity(self):
        self.assertEqual(sequence.perplexity([1]), 1.0)
        self.assertEqual(sequence.perplexity([1, 1]), 1.0)

        self.assertEqual(sequence.perplexity([0.5]), math.pow(0.5, -1 / 1.0))
        self.assertEqual(sequence.perplexity([1, 0.5]), math.pow(0.5, -1 / 2.0))
        self.assertEqual(sequence.perplexity([0.5, 0.5, 0.5]), math.pow(0.5**3, -1 / 3.0))

    def test_perplexity_zero(self):
        self.assertAlmostEqual(sequence.perplexity([0]), 1.0000000000000167e+100)

