
import collections
import heapq
import json
import logging
import math
import numpy as np
import os
import pdb

from bottle import api
from bottle import util
from pytils import check


class TrainingParameters:
    DEFAULT_BATCH_SIZE = 32
    DEFAULT_EPOCH_SIZE = 10
    DEFAULT_DROPOUT_RATE = 0.1
    DEFAULT_LEARNING_RATE = 1.0
    DEFAULT_CLIP_NORM = 5.0

    def __init__(self):
        self.batch_size = TrainingParameters.DEFAULT_BATCH_SIZE
        self.epoch_size = TrainingParameters.DEFAULT_EPOCH_SIZE
        self.dropout_rate = TrainingParameters.DEFAULT_DROPOUT_RATE
        self.learning_rate = TrainingParameters.DEFAULT_LEARNING_RATE
        self.clip_norm = TrainingParameters.DEFAULT_CLIP_NORM

    def _copy(self, override_key, override_value):
        clss = type(self)
        copied = clss.__new__(clss)
        copied.__dict__ = {k: v for k, v in self.__dict__.items()}
        copied.__dict__[override_key] = override_value
        return copied

    def with_batch_size(self, value):
        return self._copy("batch_size", check.check_gte(value, 1))

    def with_epoch_size(self, value):
        return self._copy("epoch_size", check.check_gte(value, 1))

    def with_dropout_rate(self, value=None):
        return self._copy("dropout_rate", check.check_gte(value, 0))

    def with_learning_rate(self, value=None):
        return self._copy("learning_rate", check.check_gte(value, 0))

    def with_clip_norm(self, value=None):
        return self._copy("clip_norm", check.check_gt(value, 0))

    def __repr__(self):
        return "TrainingParameters{batch=%d, epoch=%d, dropout=%.4f, learning=%.4f, clip=%.4f}" % \
            (self.batch_size, self.epoch_size, self.dropout_rate, self.learning_rate, self.clip_norm)


class ProgressMarker:
    def __init__(self, improved, update_best):
        self.improved = check.check_one_of(improved, [True, False])
        self.update_best = check.check_one_of(update_best, [True, False])


class TrainingSchedule:
    DEFAULT_DECAY_RATE = 0.85
    DEFAULT_MAXIMUM_EPOCHS = TrainingParameters.DEFAULT_EPOCH_SIZE * 10
    DEFAULT_MAXIMUM_DECAYS = 15
    DEFAULT_WINDOW_SIZE = 10
    DEFAULT_GROWTH_RATE = 0.01
    DEFAULT_LENIENCY_RATE = 0.001
    REASON_MAXIMUM_EPOCHS = "maximum (epochs=%d, threshold=%d)"
    REASON_MAXIMUM_DECAYS = "maximum (decays=%d, threshold=%d)"

    def __init__(self):
        self.decay_rate = TrainingSchedule.DEFAULT_DECAY_RATE
        self.maximum_epochs = TrainingSchedule.DEFAULT_MAXIMUM_EPOCHS
        self.maximum_decays = TrainingSchedule.DEFAULT_MAXIMUM_DECAYS
        self.window_size = TrainingSchedule.DEFAULT_WINDOW_SIZE
        self.growth_rate = TrainingSchedule.DEFAULT_GROWTH_RATE
        self.leniency_rate = TrainingSchedule.DEFAULT_LENIENCY_RATE

    def _copy(self, override_key, override_value):
        clss = type(self)
        copied = clss.__new__(clss)
        copied.__dict__ = {k: v for k, v in self.__dict__.items()}
        copied.__dict__[override_key] = override_value
        return copied

    def with_decay_rate(self, value):
        return self._copy("decay_rate", check.check_range(value, 0, 1))

    def with_maximum_epochs(self, value):
        return self._copy("maximum_epochs", check.check_gte(value, 1))

    def with_maximum_decays(self, value):
        return self._copy("maximum_decays", check.check_gte(value, 1))

    def with_window_size(self, value):
        return self._copy("window_size", check.check_gte(value, 1))

    def with_growth_rate(self, value):
        return self._copy("growth_rate", check.check_range(value, 0, 1))

    def with_leniency_rate(self, value):
        return self._copy("leniency_rate", check.check_range(value, 0, 1))

    def evaluate_progress(self, train_losses, best_score, current_score):
        # Based off train data.
        average_slope = util.average_slope(train_losses)
        growth_slope = -self.growth_rate
        loss_improved = average_slope <= growth_slope

        # Based off validate data.
        previous_score = best_score
        update_best = previous_score < current_score
        evaluation_score = previous_score * (1.0 + self.leniency_rate)
        validate_improved = evaluation_score < current_score

        # We've improved if the losses point downwards, or if the validate score got better (allowing for some degree of leniency).
        logging.debug("(%.4f <= %.4f) -> %s || (%.4f < %.4f) -> %s" % \
            (average_slope, growth_slope, loss_improved, evaluation_score, current_score, validate_improved))
        return ProgressMarker(loss_improved or validate_improved, update_best)

    def decay(self, train_account, training_parameters):
        lr = training_parameters.learning_rate
        return training_parameters.with_learning_rate(lr * self.decay_rate)

    def is_finished(self, train_account):
        if train_account.epoch_count >= self.maximum_epochs:
            return True, TrainingSchedule.REASON_MAXIMUM_EPOCHS % \
                (train_account.epoch_count, self.maximum_epochs)

        if train_account.decay_count >= self.maximum_decays:
            return True, TrainingSchedule.REASON_MAXIMUM_DECAYS % \
                (train_account.decay_count, self.maximum_decays)

        return False, None


class MinimumLearningSchedule(TrainingSchedule):
    DEFAULT_MINIMUM_LEARNING_RATE = 0.25
    REASON_MINIMAL_LEARNING = "minimal learning (learning rate=%f, threshold=%f)"

    def __init__(self):
        super().__init__()
        self.minimum_learning_rate = MinimumLearningSchedule.DEFAULT_MINIMUM_LEARNING_RATE

    def with_minimum_learning_rate(self, value):
        return self._copy("minimum_learning_rate", check.check_gt(value, 0))

    def is_finished(self, train_account):
        if train_account.decayed_learning_rate < self.minimum_learning_rate:
            return True, MinimumLearningSchedule.REASON_MINIMAL_LEARNING % \
                (train_account.decayed_learning_rate, self.minimum_learning_rate)

        return super().is_finished(train_account)


class ConvergingSchedule(TrainingSchedule):
    DEFAULT_CONVERGED_RATE = 0.005
    REASON_CONVERGED = "converged (average slopes=%s, threshold slope=%f)"

    def __init__(self):
        super().__init__()
        # The downward slope of the rate of loss at which to consider the training converged.
        # Consider the converged rate of 0.05, which maps to the converged slope -0.05 and linear function `y = -0.05x`.
        # Also, imagine producing a similar linear function `y' = AVERAGE_SLOPEx` from the actual losses.
        # Then, when AVERAGE_SLOPE > -0.05, or visually when y' goes below y consistently across the window, then we have converged.
        self.converged_rate = ConvergingSchedule.DEFAULT_CONVERGED_RATE
        self.average_slopes = []

    def with_converged_rate(self, value):
        return self._copy("converged_rate", check.check_gt(value, 0))

    def is_finished(self, train_account):
        if train_account.loss_window.is_full():
            average_slope = util.average_slope(train_account.loss_window)
            converged_slope = -self.converged_rate
            logging.debug("(%.4f > %.4f) -> %s" % (average_slope, converged_slope, average_slope > converged_slope))

            if average_slope > converged_slope:
                return True, ConvergingSchedule.REASON_CONVERGED % \
                    (average_slope, converged_slope)

        return super().is_finished(train_account)


class TrainAccount:
    def __init__(self, window_size):
        self.loss_window = util.Window(window_size)
        self.epoch_count = 0
        self.decay_count = 0
        self.decayed_learning_rate = None
        self.version = 0
        self.best_score = None

    def baseline(self, baseline_score):
        self.best_score = baseline_score

    def record_round(self, round_losses, score, progress_marker):
        if self.best_score is None:
            raise ValueError("Baseline must be set before recording can begin.")

        self.loss_window.append_all(round_losses)
        self.epoch_count += len(round_losses)
        self.version += 1

        if progress_marker.update_best:
            self.best_score = score

    def record_decay(self, decayed_learning_rate):
        if self.best_score is None:
            raise ValueError("Baseline must be set before recording can begin.")

        self.decay_count += 1
        self.decayed_learning_rate = decayed_learning_rate


class TrainingHarness:
    def __init__(self, parameters, schedule):
        self.parameters = check.check_instance(parameters, TrainingParameters)
        self.schedule = check.check_instance(schedule, TrainingSchedule)

    def train(self, model_persistence, dataset, debug=False):
        check.check_instance(model_persistence, api.model.ModelPersistence)
        check.check_instance(model_persistence.model, api.model.IterativelyOptimized)
        check.check_instance(dataset, api.data.Dataset)
        train_account = TrainAccount(self.schedule.window_size)
        score = model_persistence.model.score(dataset.validate)
        train_account.baseline(score)
        logging.debug("Baseline validate score: %.4f" % (score))
        model_persistence.save(train_account.version, {"score_validate": score})
        training_parameters = self.parameters

        if debug:
            logging.debug("Training under: %s." % training_parameters)

        while True:
            finished, reason = self.schedule.is_finished(train_account)

            if finished:
                assert reason is not None, "when the schedule is finished it must provide a reason"
                logging.debug("Finished training: %s" % reason)
                break

            round_losses = self._optimization_round(model_persistence.model, dataset.train, training_parameters, debug)
            score = model_persistence.model.score(dataset.validate)
            progress_marker = self.schedule.evaluate_progress(round_losses, train_account.best_score, score)

            if progress_marker.improved:
                logging.debug("Progress improved        - proceeding.  Validate scores: previous=%.4f, current=%.4f." % \
                    (train_account.best_score, score))
                train_account.record_round(round_losses, score, progress_marker)
                model_persistence.save(train_account.version, {"score_validate": score})
            else:
                logging.debug("Progress did not improve -   decaying.  Validate scores: previous=%.4f, current=%.4f." % \
                    (train_account.best_score, score))
                train_account.record_decay(training_parameters.learning_rate)
                model_persistence.load(train_account.version)
                training_parameters = self.schedule.decay(train_account, training_parameters)

                if debug:
                    logging.debug("Training under: %s." % training_parameters)

        score_train = model_persistence.model.score(dataset.train)
        score_test = model_persistence.model.score(dataset.test)
        logging.debug("Final train / test scores: %.4f / %.4f" % (score_train, score_test))

    def _optimization_round(self, model, trainstream, training_parameters, debug):
        check.check_instance(model, api.model.IterativelyOptimized)
        check.check_instance(trainstream, api.data.Datastream)
        check.check_instance(training_parameters, api.train.TrainingParameters)
        model_parameters = model.extract_parameters(training_parameters)
        randomized_trainstream = trainstream.as_randomized(training_parameters.batch_size * 4)
        slot_length = util.order_of_magnitude(training_parameters.epoch_size)
        epoch_template = "Epoch {:%dd} loss: {:.6f}" % slot_length
        epoch = -1
        losses = []

        while epoch + 1 < training_parameters.epoch_size:
            epoch += 1
            epoch_loss = model.step_optimize(model_parameters, randomized_trainstream, training_parameters.batch_size)
            losses += [epoch_loss]

            if debug:
                logging.debug(epoch_template.format(epoch, epoch_loss))

        return losses


