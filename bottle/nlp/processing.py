
import nltk.tokenize
import pdb
import re

from pytils import check

from bottle.nlp.model import Token, SentenceBuilder


def is_number(word):
    try:
        float(word.replace(",", ""))
        return True
    except ValueError:
        return False


def is_reserved_token(word):
    return re.match("<\w+>", word) is not None


def word_tokens(text_or_stream):
    def tokenize(text):
        hold_back = None
        skip = False

        for word in nltk.tokenize.word_tokenize(text):
            if hold_back is not None:
                if word == hold_back[0]:
                    yield Token(hold_back[0])
                    yield Token(hold_back[1])
                    yield Token(word)
                    skip = True
                else:
                    yield Token(hold_back[0] + hold_back[1])

                hold_back = None

            if not skip:
                if word.startswith(Token.APOSTROPHE):
                    # Use hold_back to fix tokenization errors of the form:
                    # | input  | output  | expected |
                    # | ------ | ------- | -------- |
                    # | 'word' | 'word ' | ' word ' |
                    hold_back = (word[0], word[1:])
                else:
                    hold_back = None

                if hold_back is None:
                    yield Token(word)

            skip = False

        if hold_back is not None:
            yield Token(hold_back[0] + hold_back[1])

    if isinstance(text_or_stream, str):
        for token in tokenize(text_or_stream):
            yield token
    else:
        for text in text_or_stream:
            for token in tokenize(text):
                yield token


def as_sentence(words):
    sentence_builder = SentenceBuilder()

    for i, word in enumerate(words):
        if isinstance(word, Token):
            token = word
        else:
            token = Token(word)

        # We can only complete a sentence when we're at the final token.
        #                               v
        sentence_builder.process(token, can_complete=i + 1 == len(words))

    return sentence_builder.build()


def sentences(word_token_stream):
    sentence_builder = SentenceBuilder()

    for token in word_token_stream:
        check.check_instance(token, Token)

        # Since we are streaming words in, we can complete the sentence at any time.
        #                                  v
        if sentence_builder.process(token, can_complete=True):
            yield sentence_builder.build()
            sentence_builder = SentenceBuilder()

    if not sentence_builder.is_empty():
        yield sentence_builder.build()

