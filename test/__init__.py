
from pytils.invigilator import create_suite


import test.api.data
import test.nlp.dictionary
import test.nlp.processing
import test.util


def all():
    return create_suite(unit())


def unit():
    return [
        test.api.data.tests(),
        test.nlp.dictionary.tests(),
        test.nlp.processing.tests(),
        test.util.tests(),
    ]

