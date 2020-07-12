
import collections
import io
import nltk.tokenize
import pdb

from pytils import adjutant, check


# Tokens
START = "<start>"
END = "<end>"
UNKNOWN = "<unknown>"
NUMBER = "<number>"


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
                if word.startswith("'"):
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


def sentences(word_token_stream):
    history = collections.deque(maxlen=5)
    open_close_stack = []
    quoted = False
    complete = False
    sentence = []

    for token in word_token_stream:
        check.check_instance(token, Token)
        history.append(token)

        if token.is_quote():
            if quoted:
                quoted = False
            else:
                quoted = True

        if token.is_open() or (token.is_quote() and quoted):
            open_close_stack += [token]
        elif token.is_close() or (token.is_quote() and not quoted):
            try:
                open_symbol = open_close_stack.pop()
            except IndexError:
                raise ValueError("Un-paired close symbol for (snippet: %s): %s" % ([i.word for i in history], token.word))

            if not open_symbol.pairs_to(token):
                raise ValueError("Un-paired open/close symbol for (snippet: %s): %s %s" % ([i.word for i in history], open_symbol.word, token.word))

        if complete:
            # If the token terminates the ready sentence.
            if len(open_close_stack) == 0:
                yield sentence + [token.literal]
                sentence = []
                complete = False
            # Otherwise, if the token makes progress towards terminating the ready sentence.
            elif token.is_terminal() or token.is_close():
                sentence += [token.literal]
            else:
                raise ValueError("Non-terminated sentence: %s" % sentence)
        else:
            sentence += [token.literal]

        if token.is_terminal():
            if len(open_close_stack) == 0:
                yield sentence
                sentence = []
                complete = False
            else:
                # We're ready to terminate, but an open_close is holding the sentence up.
                complete = True

    if complete and len(open_close_stack) == 0:
        yield sentence
    elif len(sentence) > 0:
        raise ValueError("Non-terminated sentence: %s" % sentence)


class Token:
    OPEN_SYMBOLS = {
        "``": "''",
        "[": "]",
        "(": ")",
    }
    CLOSE_SYMBOLS = adjutant.dict_invert(OPEN_SYMBOLS)
    QUOTE = '"'
    SINGLE_QUOTE = "'"
    TERMINAL_SYMBOLS = {
        ".": True,
        "?": True,
        "!": True,
    }

    def __init__(self, word):
        self.word = check.check_not_empty(word)
        self.literal = canonicalize_word(word)

    def is_open(self):
        return self.literal in Token.OPEN_SYMBOLS

    def is_close(self):
        return self.literal in Token.CLOSE_SYMBOLS

    def is_quote(self):
        return self.literal == Token.QUOTE or self.literal == Token.SINGLE_QUOTE

    def is_terminal(self):
        return self.literal in Token.TERMINAL_SYMBOLS

    def pairs_to(self, other):
        if self.is_open():
            return Token.OPEN_SYMBOLS[self.literal] == other.literal
        elif self.is_close():
            return Token.CLOSE_SYMBOLS[self.literal] == other.literal
        elif self.is_quote():
            return self.literal == other.literal

        return False

    def __repr__(self):
        return "Token{word=%s literal=%s}" % (self.word, self.literal)


def canonicalize_word(word):
    canonicalization = []

    for c in word.lower():
        c_fixed = CHARACTER_CANONICALIZATIONS[c] if c in CHARACTER_CANONICALIZATIONS else c
        canonicalization += [c_fixed]

        if not is_valid_ascii(c_fixed):
            raise ValueError("Word '%s' contains invalid character '%s'." % (word, c))

    return "".join(canonicalization)


def is_valid_ascii(character):
    decimal = ord(character)
    # first ascii character: ' ' -> 32
    # last ascii character:  '~' -> 126
    return decimal >= 32 and decimal <= 126


CHARACTER_CANONICALIZATIONS = {
    "“": "``",
    "”": "''",
    "‘": "'",
    "’": "'",
    "–": "-",

    "à": "a",
    "á": "a",
    "â": "a",
    "ä": "a",
    "æ": "a",
    "ã": "a",
    "å": "a",
    "ā": "a",
    "ǎ": "a",
    "ç": "c",
    "ć": "c",
    "č": "c",
    "è": "e",
    "é": "e",
    "ê": "e",
    "ë": "e",
    "ē": "e",
    "ė": "e",
    "ę": "e",
    "ě": "e",
    "î": "i",
    "ï": "i",
    "í": "i",
    "ī": "i",
    "į": "i",
    "ì": "i",
    "ł": "l",
    "ñ": "n",
    "ń": "n",
    "ô": "o",
    "ö": "o",
    "ò": "o",
    "ó": "o",
    "œ": "o",
    "ø": "o",
    "ō": "o",
    "õ": "o",
    "ß": "s",
    "ś": "s",
    "š": "s",
    "û": "u",
    "ü": "u",
    "ù": "u",
    "ú": "u",
    "ū": "u",
    "ÿ": "y",
    "ž": "z",
    "ź": "z",
    "ż": "z",
}

