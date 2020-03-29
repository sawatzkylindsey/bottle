
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

    def save(self, version, extra={}):
        savepoint = self.savepoints.get(version)
        logging.debug("Saving model at %s." % (savepoint))
        self.model.save_parameters(savepoint)
        self.savepoints.update(savepoint, extra, True) \
            .save()

    def load(self, version=None):
        savepoint = self.savepoints.get(version)
        logging.debug("Restoring model at %s." % savepoint)
        self.model.load_parameters(savepoint)


class ModelArchitecture:
    def get_states(self):
        raise NotImplementedError()

    def get_state(self, name, layer=None):
        raise NotImplementedError()


class State:
    def __init__(self, name, layer, width):
        self.name = check.check_not_empty(name)
        self.layer = layer

        if layer is not None:
            check.check_gte(layer, 0)

        self.width = check.check_gte(width, 1)

    def __repr__(self):
        layer_str = "" if self.layer is None else (":%d" % self.layer)
        return "State{%s%s, %d}" % (self.name, layer_str, self.width)


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
    KEY_VERSION = "version"
    KEY_EXTRA = "extra"

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
            step = self.versions[key][Savepoints.KEY_VERSION]
        except KeyError:
            step = self.next_step

        return Savepoint(self.model_dir, step, key)

    def update(self, savepoint, extra, set_latest=False):
        self.versions[savepoint.version_key] = {
            Savepoints.KEY_VERSION: savepoint.step,
            Savepoints.KEY_EXTRA: extra
        }
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
                steps = [step for step in map(lambda item: item[Savepoints.KEY_VERSION], data["versions"].values())]
                next_step = sorted(steps)[-1] if len(steps) > 0 else 0
                return Savepoints(save_dir, data["versions"], data["latest"], next_step)
            except KeyError:
                return None

