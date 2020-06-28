
from pytils.invigilator import create_suite


import test.nlp


def all():
    return create_suite(unit())


def unit():
    return [
        test.nlp.tests()
    ]

