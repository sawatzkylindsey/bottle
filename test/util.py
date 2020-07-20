
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle import util


def tests():
    return create_suite([UtilTests])


class UtilTests(TestCase):
    def test_average_slope(self):
        self.assertEqual(util.average_slope([1, 1]), 0)
        self.assertEqual(util.average_slope([1, 2]), 1)
        self.assertEqual(util.average_slope([1, 0]), -1)

        self.assertEqual(util.average_slope([100, 100]), 0)
        self.assertEqual(util.average_slope([100, 200]), 100)
        self.assertEqual(util.average_slope([100, 0]), -100)

        self.assertEqual(util.average_slope([1, 1, 1]), 0)
        self.assertEqual(util.average_slope([1, 2, 3]), 1)
        self.assertEqual(util.average_slope([1, 0, -1]), -1)

