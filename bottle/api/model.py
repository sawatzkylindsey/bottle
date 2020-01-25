
import json
import logging
import os
import re

# Intentionally breaking away from typical lexical import order.
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import tensorflow as tf
tf.logging.set_verbosity(logging.WARN)

from bottle import api
from pytils import check


class Model:
    def score(self, datastream):
        raise NotImplementedError()

    def save_parameters(self, savepoint):
        raise NotImplementedError()

    def load_parameters(self, savepoint):
        raise NotImplementedError()


class IterativelyOptimized(Model):
    def extract_parameters(self, training_parameters):
        check.check_instance(training_parameters, api.train.TrainingParameters)
        raise NotImplementedError()

    def step_optimize(self, model_paramters, datastream, batch_size):
        raise NotImplementedError()


class TfModel(Model):
    def __init__(self):
        super().__init__()
        self.namespace = None
        self.save_name = None
        self.saver = None

    def assert_shape(self, tensor, expected):
        assert tensor.shape.as_list() == expected, "actual %s != expected %s" % (tensor.shape, expected)

    def define_in_namespace(self, namespace, computational_graph_definition):
        if self.namespace is not None:
            raise ValueError("model may only be initialized once.")

        self.namespace = check.check_not_empty(namespace)
        self.save_name = check.check_not_empty(re.sub("[^a-zA-Z0-9]", "", self.namespace))

        with tf.variable_scope(self.namespace):
            logging.debug("Defining computational graph.")
            computational_graph_definition()

        self.saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.namespace))
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        session.run(tf.global_variables_initializer())
        logging.debug("Defined computational graph and initialized session.")
        return session

    def placeholder(self, name, shape, dtype=tf.float32):
        return tf.placeholder(dtype, shape, name=name)

    def variable(self, name, shape, dtype=tf.float32, initial=None):
        return tf.get_variable(name, shape=shape, dtype=dtype,
            initializer=tf.contrib.layers.xavier_initializer() if initial is None else tf.constant_initializer(initial))

    def save_parameters(self, savepoint):
        self.saver.save(self.session, os.path.join(savepoint.model_dir, self.save_name), global_step=savepoint.step)

    def load_parameters(self, savepoint):
        self.saver.restore(self.session, "%s-%d" % (os.path.join(savepoint.model_dir, self.save_name), savepoint.step))


class ModelPersistence:
    def __init__(self, model, save_dir):
        self.model = check.check_instance(model, Model)
        self.savepoints = Savepoints.load(save_dir)

        if self.savepoints is None:
            self.savepoints = Savepoints(save_dir)
            os.makedirs(save_dir, exist_ok=True)

    def save(self, version):
        savepoint = self.savepoints.get(version)
        logging.debug("Saving model at %s." % (savepoint))
        self.model.save_parameters(savepoint)
        self.savepoints.update(savepoint, True) \
            .save()

    def load(self, version=None):
        savepoint = self.savepoints.get(version)
        logging.debug("Restoring model at %s." % savepoint)
        self.model.load_parameters(savepoint)


class Result:
    def __init__(self, labels, array):
        self.labels = check.check_instance(labels, Labels)
        self.array = array
        self._prediction = None
        self._distribution = None
        self._ranked_items = None
        self._rank_cache = {}

    def prediction(self):
        if self._prediction is None:
            self._prediction = self.labels.vector_decode(self.array)

        return self._prediction

    def distribution(self):
        if self._distribution is None:
            self._distribution = self.labels.vector_decode_distribution(self.array)

        return self._distribution

    def rank_of(self, value, handle_unknown=False, k=None):
        target = self.labels.decode(self.labels.encode(value, handle_unknown))

        if partial_sort_off:
            if self._ranked_items is None:
                self._ranked_items = {item[0]: rank for rank, item in enumerate(sorted(self.distribution().items(), key=lambda item: item[1], reverse=True))}

            return self._ranked_items[target]

        # Use a partial sort to find the rank.
        distribution_items = [item for item in self.distribution().items()]

        # Partial sort mechanism 'nlargest'.
        if k is not None:
            if self._ranked_items is None or len(self._ranked_items) < k:
                largest = heapq.nlargest(k, distribution_items, key=lambda item: item[1])
                self._ranked_items = {item[0]: rank for rank, item in enumerate(sorted(largest, key=lambda item: item[1], reverse=True))}

            if target in self._ranked_items:
                return self._ranked_items[target]
            else:
                # This is a lie - the nlargest method says the correct rank of the top-k elements, after
                # which everything else is given the last rank.
                return len(self.labels) - 1

        # Partial sort mechanism 'insertion-sort'.
        if target in self._rank_cache:
            return self._rank_cache[target]

        insertion_sorted = []
        partial_index = 0
        partial_total = 0
        index = 0
        rank = None

        while rank is None:
            encoding, probability = distribution_items[index]
            partial_total += probability
            insertion_index = binary_search(insertion_sorted, probability, accessor=lambda item: item[1])
            insertion_sorted.insert(insertion_index, (encoding, probability))

            if len(insertion_sorted) == len(distribution_items):
                # The entire list of items have been insertion sorted.
                # Find the target.
                while rank is None:
                    if partial_index >= len(insertion_sorted):
                        logging.info("something wrong for value '%s' target '%s' (handle unknown %s)" % (value, target, handle_unknown))

                    if insertion_sorted[partial_index][0] == target:
                        rank = partial_index

                    partial_index += 1
            else:
                # A new item has been insertion sorted.
                # Move up the partial index as much as is possible.
                remaining = 1.0 - partial_total

                # We can move up the partial index as long as the sum of the unknown portion of the probability distribution is less than
                # the probability at the current partial index.
                # This is true because we know that none of the unknown probabilities would exceed it (in which case they would need to be
                # insertion sorted in a way that changes the item at the partial index's rank).
                while partial_index < len(insertion_sorted) and remaining < insertion_sorted[partial_index][1]:
                    if insertion_sorted[partial_index][0] == target:
                        rank = partial_index
                        break

                    partial_index += 1

            index += 1

        self._rank_cache[target] = rank
        return self._rank_cache[target]

    def __repr__(self):
        return "(.., prediction=%s)" % (self.prediction())


class Savepoint:
    def __init__(self, model_dir, step, version_key):
        self.model_dir = check.check_instance(model_dir, str)
        self.step = check.check_gte(check.check_instance(step, int), 0)
        self.version_key = check.check_instance(version_key, str)

    def __repr__(self):
        return "Savepoint{%s, %d, %s}" % (self.model_dir, self.step, self.version_key)


class Savepoints:
    SAVEPOINTS_FILE = "savepoints.json"
    MODEL_DIR = "model"

    def __init__(self, save_dir, versions={}, latest=None, next_step=0):
        self.save_dir = check.check_instance(save_dir, str)
        self.savepoints_file = os.path.join(self.save_dir, Savepoints.SAVEPOINTS_FILE)
        self.model_dir = os.path.join(self.save_dir, Savepoints.MODEL_DIR)
        self.versions = check.check_instance(versions, dict)
        self.latest = latest
        self.next_step = next_step

    def version_key(self, version):
        return "v%s" % str(version)

    def get(self, version=None):
        if version is None:
            key = self.latest
        else:
            key = self.version_key(version)

        try:
            step = self.versions[key]
        except KeyError:
            step = self.next_step

        return Savepoint(self.model_dir, step, key)

    def update(self, savepoint, set_latest=False):
        self.versions[savepoint.version_key] = savepoint.step
        self.next_step += 1

        if self.latest is None or set_latest:
            self.latest = savepoint.version_key

        return self

    def as_json(self):
        return {
            "versions": self.versions,
            "latest": self.latest,
        }

    def save(self):
        os.makedirs(self.save_dir, exist_ok=True)

        with open(self.savepoints_file, "w") as fh:
            json.dump(self.as_json(), fh)

    @classmethod
    def load(self, save_dir):
        if os.path.isfile(save_dir) or (save_dir.endswith("/") and os.path.isfile(os.path.dirname(save_dir))):
            raise ValueError("save_dir '%s' must not be a file." % save_dir)

        savepoints_file = os.path.join(save_dir, Savepoints.SAVEPOINTS_FILE)

        if not os.path.exists(savepoints_file):
            return None

        with open(savepoints_file, "r") as fh:
            try:
                data = json.load(fh)
            except json.decoder.JSONDecodeError:
                return None

            try:
                steps = [step for step in data["versions"].values()]
                next_step = sorted(steps)[-1] if len(steps) > 0 else 0
                return Savepoints(save_dir, data["versions"], data["latest"], next_step)
            except KeyError:
                return None

