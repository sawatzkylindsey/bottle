
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle.nlp import processing


def tests():
    return create_suite([TokenTests, StrictTokenTests, WordTokensTests, SentencesTests])


class TokenTests(TestCase):
    def test_token(self):
        value = "MyWord"
        token = processing.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.literal, value.lower())

    def test_apostrophe(self):
        value = "'"
        token = processing.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_open(), False)
        self.assertEqual(token.is_close(), False)
        self.assertEqual(token.is_quote(), False)
        self.assertEqual(token.is_apostrophe(), True)

        value = "'s"
        token = processing.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

        value = "s'"
        token = processing.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

        value = "'s'"
        token = processing.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

        value = "wo'rd"
        token = processing.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_apostrophe(), False)

    def test_quote(self):
        value = '"'
        token = processing.Token(value)
        self.assertEqual(token.word, value)
        self.assertEqual(token.is_open(), False)
        self.assertEqual(token.is_close(), False)
        self.assertEqual(token.is_quote(), True)

        value = '"s'
        token = processing.Token(value)
        self.assertEqual(token.word, value)

        value = 's"'
        token = processing.Token(value)
        self.assertEqual(token.word, value)

        value = '"s"'
        token = processing.Token(value)
        self.assertEqual(token.word, value)

        value = 'wo"rd'
        token = processing.Token(value)
        self.assertEqual(token.word, value)

    def test_canonicalization(self):
        token = processing.Token("Über")
        self.assertEqual(token.literal, "uber")

        token = processing.Token("łeet")
        self.assertEqual(token.literal, "leet")

    def test_canonicalization_strict(self):
        token = processing.Token("€")
        self.assertEqual(token.literal, "€")

        token = processing.Token("â€s")
        self.assertEqual(token.literal, "a€s")


class StrictTokenTests(TestCase):
    def setUp(self):
        processing.activate_strict()

    def tearDown(self):
        processing.deactivate_strict()

    def test_canonicalization_strict(self):
        with self.assertRaisesRegex(ValueError, "invalid character"):
            processing.Token("€")

        with self.assertRaisesRegex(ValueError, "invalid character"):
            processing.Token("â€s")


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
        sentence = "i eat food .".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        # Terminal ?
        sentence = "i eat food ?".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        # Terminal !
        sentence = "i eat food !".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

    def test_multiple(self):
        generator = processing.sentences([processing.Token(w) for w in "i eat food . you ate pie .".split()])
        self.assertEqual([s for s in generator], [
            "i eat food .".split(" "),
            "you ate pie .".split(" "),
        ])

    def test_symbols(self):
        generator = processing.sentences([processing.Token(w) for w in "i - eat { food < .".split()])
        self.assertEqual([s for s in generator], ["i - eat { food < .".split(" ")])

        generator = processing.sentences([processing.Token(w) for w in "i + eat } food > .".split()])
        self.assertEqual([s for s in generator], ["i + eat } food > .".split(" ")])

        generator = processing.sentences([processing.Token(w) for w in "i / eat | food \ .".split()])
        self.assertEqual([s for s in generator], ["i / eat | food \ .".split(" ")])

        generator = processing.sentences([processing.Token(w) for w in "i ` eat ~ food @ .".split()])
        self.assertEqual([s for s in generator], ["i ` eat ~ food @ .".split(" ")])

        generator = processing.sentences([processing.Token(w) for w in "i # eat $ food % .".split()])
        self.assertEqual([s for s in generator], ["i # eat $ food % .".split(" ")])

        generator = processing.sentences([processing.Token(w) for w in "i ^ eat & food * .".split()])
        self.assertEqual([s for s in generator], ["i ^ eat & food * .".split(" ")])

        generator = processing.sentences([processing.Token(w) for w in "i 't eat : food ; .".split()])
        self.assertEqual([s for s in generator], ["i 't eat : food ; .".split(" ")])

        generator = processing.sentences([processing.Token(w) for w in "i , eat , food * .".split()])
        self.assertEqual([s for s in generator], ["i , eat , food * .".split(" ")])

    def test_apostrophe(self):
        sentence = "the teachers ' students are \" junior congress members \" .".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

    def test_not_terminated(self):
        sentence = "i eat".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i eat `` food".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i eat `` food ''".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i eat `` food \"".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

    def test_not_closed(self):
        # Determined by abrupt non-termination
        sentence = "i \" eat .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i `` eat .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i ( eat .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i [ eat .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        # Determined by continuing non-termination
        sentence = "i ' eat . nothing".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i \" eat . nothing".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i `` eat . nothing".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i ( eat . nothing".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i [ eat . nothing".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Non-terminated"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        # Determined by other closing.
        sentence = "i \" eat '' .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i `` eat ] .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i ( eat '' .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i [ eat ) .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Un-paired open/close"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

    def test_not_opened(self):
        sentence = "i '' eat .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i ) eat .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

        sentence = "i ] eat .".split()
        self.assertFalse(processing.is_complete_sentence(sentence))
        with self.assertRaisesRegex(ValueError, "Un-paired close"):
            [s for s in processing.sentences([processing.Token(w) for w in sentence])]

    def test_quoted(self):
        sentence = "i eat ' food ' .".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        sentence = "i eat ' food . '".split()
        self.assertFalse(processing.is_complete_sentence(sentence))

        sentence = "'".split()
        self.assertFalse(processing.is_complete_sentence(sentence))

        sentence = "i eat ' food .".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        sentence = "i eat \" food \" .".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        sentence = "i eat \" food . \"".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

    def test_quotation(self):
        # Terminal .
        sentence = "i eat `` food '' .".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        sentence = "i eat `` food . ''".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        # Terminal ?
        sentence = "i eat `` food '' ?".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        sentence = "i eat `` food ? ''".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        # Terminal !
        sentence = "i eat `` food '' !".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

        sentence = "i eat `` food ! ''".split()
        self.assertTrue(processing.is_complete_sentence(sentence))
        generator = processing.sentences([processing.Token(w) for w in sentence])
        self.assertEqual([s for s in generator], [sentence])

    def test_eventually_terminal(self):
        generator = processing.sentences([processing.Token(w) for w in "i eat `` ( [ ( `` food . '' ) ] ) ''".split()])
        self.assertEqual([s for s in generator], ["i eat `` ( [ ( `` food . '' ) ] ) ''".split(" ")])

