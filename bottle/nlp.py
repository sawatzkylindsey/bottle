
from nltk.tokenize import word_tokenize
import pdb

from pytils import check


# Tokens
START = "<start>"
END = "<end>"
UNKNOWN = "<unknown>"
NUMBER = "<number>"


def split_words(corpus):
    return [word.lower() for word in word_tokenize(corpus)]


def terminal(word):
    return word == "." or \
        word == "?" or \
        word == "!" or \
        word == "\"" or \
        word == "''"


def split_sentences(text):
    words = split_words(text)
    sentences = []
    sentence = []
    terminate = False

    for i, word in enumerate(words):
        if word == END or terminate:
            assert len(sentence) > 0
            sentences += [sentence]
            sentence = [word]
            terminate = False
        else:
            sentence += [word]
            # Terminate the sentence if this is a terminal word (.!?) and if there aren't any subsequent terminal words.
            terminate = terminal(word) and (i + 1 >= len(words) or not terminal(words[i + 1]))

            if word == "\"" or word == "''":
                terminate = i + 1 < len(words) and terminal(words[i + 1])

    if len(sentence) > 0:
        sentences += [sentence]

    return sentences


def corpus_vocabulary(corpus_lines):
    words = set()

    for line in corpus_lines:
        for word in split_words(line):
            words.add(word)

    return vocabulary(words)


def corpus_sequences(corpus_stream):
    corpus_lines = None
    total_count = 0.0
    words = set()
    xy_sequences = []

    for line in corpus_stream:
        for sentence in split_sentences(line):
            sequence = []
            total_count += len(sentence)

            for i, word in enumerate(sentence):
                words.add(word)
                sequence.append(word)

            if len(sequence) > 0:
                xy_sequences.append(sequence)

    return words, xy_sequences

