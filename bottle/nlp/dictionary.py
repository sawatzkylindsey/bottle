
from pytils import check

from bottle.nlp import processing


class Builder:
    def __init__(self, minimum_count, exclude_numbers):
        self.minimum_count = check.check_gte(minimum_count, 1)
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
        return set([word for word, count in self.word_counts.items() if count >= self.minimum_count])

