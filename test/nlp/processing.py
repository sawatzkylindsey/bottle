
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle.nlp.model import Token
from bottle.nlp import processing


def tests():
    return create_suite([WordTokensTests, SentencesTests])


class WordTokensTests(TestCase):
    def test_terminals(self):
        # Terminal .
        generator = processing.word_tokens("i eat food.")
        self.assertEqual([t.literal for t in generator], "i eat food .".split(" "))

        # Terminal ?
        generator = processing.word_tokens("i eat food?")
        self.assertEqual([t.literal for t in generator], "i eat food ?".split(" "))

        # Terminal !
        generator = processing.word_tokens("i eat food!")
        self.assertEqual([t.literal for t in generator], "i eat food !".split(" "))

    def test_whitespace(self):
        generator = processing.word_tokens(" i  eat      food\u00a0. ")
        self.assertEqual([t.literal for t in generator], "i eat food .".split(" "))

    def test_lines(self):
        generator = processing.word_tokens("i  eat\nfood\r.")
        self.assertEqual([t.literal for t in generator], "i eat food .".split(" "))

    def test_capitalization(self):
        generator = processing.word_tokens("I eAt FOOD")
        self.assertEqual([t.literal for t in generator], "i eat food".split(" "))

    def test_apostrophe(self):
        generator = processing.word_tokens("the new series ' .")
        self.assertEqual([t.literal for t in generator], "the new series ' .".split(" "))

        generator = processing.word_tokens("we don't care.")
        self.assertEqual([t.literal for t in generator], "we do n't care .".split(" "))

        generator = processing.word_tokens("we'll eat soon.")
        self.assertEqual([t.literal for t in generator], "we 'll eat soon .".split(" "))

        generator = processing.word_tokens("i eat 'food'.")
        self.assertEqual([t.literal for t in generator], "i eat ' food ' .".split(" "))

        generator = processing.word_tokens("i eat ' food'.")
        self.assertEqual([t.literal for t in generator], "i eat ' food ' .".split(" "))

        generator = processing.word_tokens("i eat 'food '.")
        self.assertEqual([t.literal for t in generator], "i eat ' food ' .".split(" "))

        generator = processing.word_tokens("i eat ' food '.")
        self.assertEqual([t.literal for t in generator], "i eat ' food ' .".split(" "))

    def test_quotation(self):
        generator = processing.word_tokens("i eat \"food\"")
        self.assertEqual([t.literal for t in generator], "i eat `` food ''".split(" "))

        generator = processing.word_tokens("i eat \"food\".")
        self.assertEqual([t.literal for t in generator], "i eat `` food '' .".split(" "))

        generator = processing.word_tokens("i eat \"food.\"")
        self.assertEqual([t.literal for t in generator], "i eat `` food . ''".split(" "))


class SentencesTests(TestCase):
    def test_terminals(self):
        # Terminal .
        words = "i eat food .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        # Terminal ?
        words = "i eat food ?".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        # Terminal !
        words = "i eat food !".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

    def test_multiple(self):
        generator = processing.sentences([processing.Token(w) for w in "i eat food . you ate pie .".split()])
        self.assertEqual([s for s in generator], [
            "i eat food .".split(" "),
            "you ate pie .".split(" "),
        ])

    def test_symbols(self):
        words = "i - eat { food < .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i + eat } food > .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i / eat | food \ .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i ` eat ~ food @ .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i # eat $ food % .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i ^ eat & food * .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i 't eat : food ; .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i , eat , food * .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

    def test_apostrophe(self):
        words = "the teachers ' students are \" junior congress members \" .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

    def test_not_terminated(self):
        words = "i eat".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i eat `` food".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i eat `` food ''".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i eat `` food \"".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

    def test_not_closed(self):
        # Determined by abrupt non-termination
        words = "i \" eat .".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i `` eat .".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i ( eat .".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i [ eat .".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        # Determined by continuing non-termination
        words = "i ' eat . nothing".split()
        with self.assertRaisesRegex(ValueError, "Early-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i \" eat . nothing".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i `` eat . nothing".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i ( eat . nothing".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i [ eat . nothing".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        # Determined by other closing.
        words = "i \" eat '' .".split()
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i `` eat ] .".split()
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i ( eat '' .".split()
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i [ eat ) .".split()
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

    def test_not_opened(self):
        words = "i '' eat .".split()
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i ) eat .".split()
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i ] eat .".split()
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

    def test_quoted(self):
        words = "i eat ' food ' .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i eat ' food . '".split()
        with self.assertRaisesRegex(ValueError, "Early-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "'".split()
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            processing.as_sentence(words)
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in words])]

        words = "i eat ' food .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i eat \" food \" .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i eat \" food . \"".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

    def test_quotation(self):
        # Terminal .
        words = "i eat `` food '' .".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i eat `` food . ''".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        # Terminal ?
        words = "i eat `` food '' ?".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i eat `` food ? ''".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        # Terminal !
        words = "i eat `` food '' !".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

        words = "i eat `` food ! ''".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

    def test_eventually_terminal(self):
        words = "i eat `` ( [ ( `` food . '' ) ] ) ''".split()
        self.assertEqual(processing.as_sentence(words), words)
        generator = processing.sentences([processing.Token(w) for w in words])
        self.assertEqual([s for s in generator], [words])

