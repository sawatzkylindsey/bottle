
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle.nlp.model import Token, activate_strict, deactivate_strict


def tests():
    return create_suite([TokenTests, StrictTokenTests])


class TokenTests(TestCase):
    def test_token(self):
        value = "MyWord"
        token = Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.literal, value.lower())

    def test_apostrophe(self):
        value = "'"
        token = Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_open(), False)
        self.assertEqual(token.is_close(), False)
        self.assertEqual(token.is_quote(), False)
        self.assertEqual(token.is_apostrophe(), True)

        value = "'s"
        token = Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

        value = "s'"
        token = Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

        value = "'s'"
        token = Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

        value = "wo'rd"
        token = Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

    def test_quote(self):
        value = '"'
        token = Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_open(), False)
        self.assertEqual(token.is_close(), False)
        self.assertEqual(token.is_quote(), True)

        value = '"s'
        token = Token(value)
        self.assertEqual(token.word, value)

        value = 's"'
        token = Token(value)
        self.assertEqual(token.word, value)

        value = '"s"'
        token = Token(value)
        self.assertEqual(token.word, value)

        value = 'wo"rd'
        token = Token(value)
        self.assertEqual(token.word, value)

    def test_canonicalization(self):
        token = Token("Über")
        self.assertEqual(token.literal, "uber")

        token = Token("łeet")
        self.assertEqual(token.literal, "leet")

    def test_canonicalization_strict(self):
        token = Token("€")
        self.assertEqual(token.literal, "€")

        token = Token("â€s")
        self.assertEqual(token.literal, "a€s")


class StrictTokenTests(TestCase):
    def setUp(self):
        activate_strict()

    def tearDown(self):
        deactivate_strict()

    def test_canonicalization_strict(self):
        with self.assertRaisesRegex(ValueError, "invalid character"):
            Token("€")

        with self.assertRaisesRegex(ValueError, "invalid character"):
            Token("â€s")


