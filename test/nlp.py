
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle import nlp


def tests():
    return create_suite([TokenTests, WordTokensTests, SentencesTests])


class TokenTests(TestCase):
    def test_token(self):
        value = "MyWord"
        token = nlp.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.literal, value.lower())

    def test_single_quote(self):
        with self.assertRaisesRegex(ValueError, "is invalid"):
            nlp.Token("'")

    def test_quote(self):
        with self.assertRaisesRegex(ValueError, "contains invalid character"):
            nlp.Token('"')

        with self.assertRaisesRegex(ValueError, "contains invalid character"):
            nlp.Token('word"')

        with self.assertRaisesRegex(ValueError, "contains invalid character"):
            nlp.Token('"word')

        with self.assertRaisesRegex(ValueError, "contains invalid character"):
            nlp.Token('wo"rd')

    def test_canonicalization(self):
        token = nlp.Token("Über")
        self.assertEqual(token.literal, "uber")

        token = nlp.Token("łeet")
        self.assertEqual(token.literal, "leet")


class WordTokensTests(TestCase):
    def test_terminals(self):
        # Terminal .
        generator = nlp.word_tokens("i eat food.")
        self.assertEqual([t.literal for t in generator], "i eat food .".split(" "))

        # Terminal ?
        generator = nlp.word_tokens("i eat food?")
        self.assertEqual([t.literal for t in generator], "i eat food ?".split(" "))

        # Terminal !
        generator = nlp.word_tokens("i eat food!")
        self.assertEqual([t.literal for t in generator], "i eat food !".split(" "))

    def test_whitespace(self):
        generator = nlp.word_tokens(" i  eat      food\u00a0. ")
        self.assertEqual([t.literal for t in generator], "i eat food .".split(" "))

    def test_lines(self):
        generator = nlp.word_tokens("i  eat\nfood\r.")
        self.assertEqual([t.literal for t in generator], "i eat food .".split(" "))

    def test_capitalization(self):
        generator = nlp.word_tokens("I eAt FOOD")
        self.assertEqual([t.literal for t in generator], "i eat food".split(" "))

    def test_single_quoted(self):
        generator = nlp.word_tokens("i eat 'food'.")
        self.assertEqual([t.literal for t in generator], "i eat 'food' .".split(" "))

    def test_quotation(self):
        generator = nlp.word_tokens("i eat \"food\"")
        self.assertEqual([t.literal for t in generator], "i eat `` food ''".split(" "))

        generator = nlp.word_tokens("i eat \"food\".")
        self.assertEqual([t.literal for t in generator], "i eat `` food '' .".split(" "))

        generator = nlp.word_tokens("i eat \"food.\"")
        self.assertEqual([t.literal for t in generator], "i eat `` food . ''".split(" "))


class SentencesTests(TestCase):
    def test_terminals(self):
        # Terminal .
        generator = nlp.sentences([nlp.Token(w) for w in "i eat food .".split()])
        self.assertEqual([s for s in generator], ["i eat food .".split(" ")])

        # Terminal ?
        generator = nlp.sentences([nlp.Token(w) for w in "i eat food ?".split()])
        self.assertEqual([s for s in generator], ["i eat food ?".split(" ")])

        # Terminal !
        generator = nlp.sentences([nlp.Token(w) for w in "i eat food !".split()])
        self.assertEqual([s for s in generator], ["i eat food !".split(" ")])

    def test_multiple(self):
        generator = nlp.sentences([nlp.Token(w) for w in "i eat food . you ate pie .".split()])
        self.assertEqual([s for s in generator], [
            "i eat food .".split(" "),
            "you ate pie .".split(" "),
        ])

    def test_symbols(self):
        generator = nlp.sentences([nlp.Token(w) for w in "i - eat { food < .".split()])
        self.assertEqual([s for s in generator], ["i - eat { food < .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i + eat } food > .".split()])
        self.assertEqual([s for s in generator], ["i + eat } food > .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i / eat | food \ .".split()])
        self.assertEqual([s for s in generator], ["i / eat | food \ .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i ` eat ~ food @ .".split()])
        self.assertEqual([s for s in generator], ["i ` eat ~ food @ .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i # eat $ food % .".split()])
        self.assertEqual([s for s in generator], ["i # eat $ food % .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i ^ eat & food * .".split()])
        self.assertEqual([s for s in generator], ["i ^ eat & food * .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i 't eat : food ; .".split()])
        self.assertEqual([s for s in generator], ["i 't eat : food ; .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i , eat , food * .".split()])
        self.assertEqual([s for s in generator], ["i , eat , food * .".split(" ")])

    def test_not_terminated(self):
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i eat".split()])]

        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i eat `` food".split()])]

        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i eat `` food ''".split()])]

        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i eat `` food ''".split()])]

    def test_not_closed(self):
        # Determined by abrupt non-termination
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i `` eat .".split()])]

        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i ( eat .".split()])]

        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i [ eat .".split()])]

        # Determined by continuing non-termination
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i `` eat . nothing".split()])]

        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i ( eat . nothing".split()])]

        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i [ eat . nothing".split()])]

        # Determined by other closing.
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i `` eat ] .".split()])]

        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i ( eat '' .".split()])]

        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i [ eat ) .".split()])]

    def test_not_opened(self):
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i '' eat .".split()])]

        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i ) eat .".split()])]

        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in nlp.sentences([nlp.Token(w) for w in "i ] eat .".split()])]

    def test_quotation(self):
        # Terminal .
        generator = nlp.sentences([nlp.Token(w) for w in "i eat `` food '' .".split()])
        self.assertEqual([s for s in generator], ["i eat `` food '' .".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i eat `` food . ''".split()])
        self.assertEqual([s for s in generator], ["i eat `` food . ''".split(" ")])

        # Terminal ?
        generator = nlp.sentences([nlp.Token(w) for w in "i eat `` food '' ?".split()])
        self.assertEqual([s for s in generator], ["i eat `` food '' ?".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i eat `` food ? ''".split()])
        self.assertEqual([s for s in generator], ["i eat `` food ? ''".split(" ")])

        # Terminal !
        generator = nlp.sentences([nlp.Token(w) for w in "i eat `` food '' !".split()])
        self.assertEqual([s for s in generator], ["i eat `` food '' !".split(" ")])

        generator = nlp.sentences([nlp.Token(w) for w in "i eat `` food ! ''".split()])
        self.assertEqual([s for s in generator], ["i eat `` food ! ''".split(" ")])

    def test_eventually_terminal(self):
        generator = nlp.sentences([nlp.Token(w) for w in "i eat `` ( [ ( `` food . '' ) ] ) ''".split()])
        self.assertEqual([s for s in generator], ["i eat `` ( [ ( `` food . '' ) ] ) ''".split(" ")])

