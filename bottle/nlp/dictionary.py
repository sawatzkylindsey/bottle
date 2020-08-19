
from pytils import check

from bottle.nlp import processing


class OccurrenceCountBuilder:
    def __init__(self, threshold, exclude_numbers):
        self.threshold = check.check_gte(threshold, 1)
        self.exclude_numbers = check.check_one_of(exclude_numbers, [True, False])
        self.word_counts = {}

    def add(self, word):
        if self.exclude_numbers and processing.is_number(word):
            return self

        if processing.is_reserved_token(word):
            return self

        if word not in self.word_counts:
            self.word_counts[word] = 0

        self.word_counts[word] += 1
        return self

    def build(self):
        return set([word for word, count in self.word_counts.items() if count >= self.threshold])


class VocabularySizeBuilder:
    def __init__(self, threshold, exclude_numbers):
        self.threshold = check.check_gte(threshold, 1)
        self.exclude_numbers = check.check_one_of(exclude_numbers, [True, False])
        self.word_counts = {}

    def add(self, word):
        if self.exclude_numbers and processing.is_number(word):
            return self

        if processing.is_reserved_token(word):
            return self

        if word not in self.word_counts:
            self.word_counts[word] = 0

        self.word_counts[word] += 1
        return self

    def build(self):
        return set([item[0] for item in sorted(self.word_counts.items(), key=lambda item: item[1], reverse=True)[:self.threshold]])

