
import bottle


class Lstm(bottle.api.model.Model):
    def __init__(self, name, input_labels, output_labels):
        super().__init__()
        self.time_dimension = None
        self.batch_dimension = None

        self.unrolled_inputs_p = self.placeholder("unrolled_inputs_p", [self.time_dimension, self.batch_dimension], tf.int32)
        self.initial_state_p = self.placeholder("initial_state_p", [SCAN_STATES, self.hyper_parameters.layers, self.batch_dimension, self.hyper_parameters.width])
        self.learning_rate_p = self.placeholder("learning_rate_p", [1], tf.float32)
