
from unittest import TestCase

from pytils.invigilator import create_suite

from bottle.nlp import dictionary


def tests():
    return create_suite([OccurrenceCountBuilderTests, VocabularySizeBuilderTests])


class OccurrenceCountBuilderTests(TestCase):
    def test_builder(self):
        word1 = "word1"
        word2 = "word2"
        word3 = "4.5"

        builder = dictionary.OccurrenceCountBuilder(1, False)
        builder.add(word1) \
            .add(word2) \
            .add(word1) \
            .add(word3)

        self.assertEqual(builder.build(), set([word1, word2, word3]))

    def test_builder_minimum(self):
        word1 = "word1"
        word2 = "word2"
        word3 = "4.5"

        builder = dictionary.OccurrenceCountBuilder(2, False)
        builder.add(word1) \
            .add(word2) \
            .add(word1) \
            .add(word3)

        self.assertEqual(builder.build(), set([word1]))

    def test_builder_exclude_numbers(self):
        builder = dictionary.OccurrenceCountBuilder(1, True)
        builder.add("23") \
            .add("-1") \
            .add("0") \
            .add("0.123") \
            .add("100,000") \
            .add("$6") \
            .add("75%")

        self.assertEqual(builder.build(), set(["$6", "75%"]))

    def test_builder_reserved_token(self):
        word1 = "word1"
        word2 = "word2"
        word3 = "4.5"

        builder = dictionary.OccurrenceCountBuilder(1, False)
        builder.add(word1) \
            .add(word2) \
            .add(word1) \
            .add(word3) \
            .add("<unk>") \
            .add("<unknown>") \
            .add("<blah>")

        self.assertEqual(builder.build(), set([word1, word2, word3]))


class VocabularySizeBuilderTests(TestCase):
    def test_builder(self):
        word1 = "word1"
        word2 = "word2"
        word3 = "4.5"

        builder = dictionary.VocabularySizeBuilder(3, False)
        builder.add(word1) \
            .add(word2) \
            .add(word1) \
            .add(word3)

        self.assertEqual(builder.build(), set([word1, word2, word3]))

    def test_builder_minimum(self):
        word1 = "word1"
        word2 = "word2"
        word3 = "4.5"

        builder = dictionary.VocabularySizeBuilder(1, False)
        builder.add(word1) \
            .add(word2) \
            .add(word1) \
            .add(word3)

        self.assertEqual(builder.build(), set([word1]))

    def test_builder_exclude_numbers(self):
        builder = dictionary.VocabularySizeBuilder(10, True)
        builder.add("23") \
            .add("-1") \
            .add("0") \
            .add("0.123") \
            .add("100,000") \
            .add("$6") \
            .add("75%")

        self.assertEqual(builder.build(), set(["$6", "75%"]))

    def test_builder_reserved_token(self):
        word1 = "word1"
        word2 = "word2"
        word3 = "4.5"

        builder = dictionary.VocabularySizeBuilder(10, False)
        builder.add(word1) \
            .add(word2) \
            .add(word1) \
            .add(word3) \
            .add("<unk>") \
            .add("<unknown>") \
            .add("<blah>")

        self.assertEqual(builder.build(), set([word1, word2, word3]))

