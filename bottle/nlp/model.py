
import collections
import os

from pytils import adjutant, check


strict_on = "bottle_strict_on" in os.environ


def activate_strict():
    global strict_on
    strict_on = True


def deactivate_strict():
    global strict_on
    strict_on = False


class Token:
    OPEN_SYMBOLS = {
        "``": "''",
        "[": "]",
        "(": ")",
    }
    CLOSE_SYMBOLS = adjutant.dict_invert(OPEN_SYMBOLS)
    QUOTE = '"'
    APOSTROPHE = "'"
    TERMINAL_SYMBOLS = {
        ".": True,
        "?": True,
        "!": True,
    }

    def __init__(self, word):
        self.word = check.check_not_empty(check.check_not_none(word))
        self.literal = canonicalize_word(word)

    def is_open(self):
        return self.literal in Token.OPEN_SYMBOLS

    def is_close(self):
        return self.literal in Token.CLOSE_SYMBOLS

    def is_quote(self):
        return self.literal == Token.QUOTE

    def is_apostrophe(self):
        return self.literal == Token.APOSTROPHE

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


class SentenceBuilder:
    def __init__(self):
        self.history = collections.deque(maxlen=5)
        self.open_close_stack = []
        self.quoted = False
        self.complete = False
        self.pending = False
        self.sentence = []

    def is_empty(self):
        return not self.complete and len(self.sentence) == 0

    def process(self, token, can_complete):
        self.history.append(token)

        if token.is_quote():
            if self.quoted:
                self.quoted = False
            else:
                self.quoted = True

        if token.is_open() or (token.is_quote() and self.quoted):
            self.open_close_stack += [token]
        elif token.is_close() or (token.is_quote() and not self.quoted):
            try:
                open_symbol = self.open_close_stack.pop()
            except IndexError:
                raise ValueError("Un-paired close symbol for (snippet: %s): %s" % ([i.word for i in self.history], token.word))

            if not open_symbol.pairs_to(token):
                raise ValueError("Un-paired open/close symbol for (snippet: %s): %s %s" % ([i.word for i in self.history], open_symbol.word, token.word))

        if self.pending:
            # If the token terminates the pending sentence.
            if len(self.open_close_stack) == 0 and can_complete:
                self.sentence += [token.literal]
                self.complete = True
                self.pending = False
            # Otherwise, if the token makes progress towards terminating the ready sentence.
            elif token.is_terminal() or token.is_close():
                self.sentence += [token.literal]
            else:
                raise ValueError("Non-terminated sentence: %s" % self.sentence)
        else:
            self.sentence += [token.literal]

        if token.is_terminal():
            if len(self.open_close_stack) == 0 and can_complete:
                self.complete = True
            elif len(self.open_close_stack) != 0:
                # We're ready to terminate, but an open_close is holding the sentence up.
                self.pending = True
            else:
                raise ValueError("Early-terminated sentence: %s" % self.sentence)

        return self.complete

    def build(self):
        if self.complete:
            return self.sentence
        else:
            raise ValueError("Non-terminated sentence: %s" % self.sentence)


def canonicalize_word(word):
    canonicalization = []

    for c in word.lower():
        c_fixed = CHARACTER_CANONICALIZATIONS[c] if c in CHARACTER_CANONICALIZATIONS else c
        canonicalization += [c_fixed]

        if strict_on and not is_valid_ascii(c_fixed):
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
    "†": "t",
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

