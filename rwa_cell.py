import collections
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops, math_ops, init_ops, nn_ops

RWACellTuple = collections.namedtuple("RWACellTuple", ("h", "n", "d", "a_max"))
_WEIGHTS_VARIABLE_NAME = rnn_cell_impl._WEIGHTS_VARIABLE_NAME
_BIAS_VARIABLE_NAME = rnn_cell_impl._BIAS_VARIABLE_NAME

class RWACell(rnn_cell_impl._LayerRNNCell):
  """Recurrent Weighted Average (cf. http://arxiv.org/abs/1703.01253)."""

  def __init__(self, num_units, activation=math_ops.tanh, normalize=False,
               bias_initializer=None, reuse=None, name=None):
    super(RWACell, self).__init__(_reuse=reuse, name=name)
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse
    self._normalize = normalize
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return RWACellTuple(self._num_units, self._num_units, self._num_units, self._num_units)

  def zero_state(self, batch_size, dtype):
    h, n, d, _ = super(RWACell, self).zero_state(batch_size, dtype)
    a_max = tf.fill([batch_size, self._num_units], -1E38) # Start off with lowest number possible
    return RWACellTuple(h, n, d, a_max)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value

    self._unbounded_kernel = self.add_variable("unbounded/%s" % _WEIGHTS_VARIABLE_NAME,
                                               shape=[input_depth, self._num_units])

    self._unbounded_bias = self.add_variable("unbounded/%s" % _BIAS_VARIABLE_NAME,
                                             shape=[self._num_units],
                                             initializer=(
                                                self._bias_initializer
                                                if self._bias_initializer is not None
                                                else init_ops.zeros_initializer(dtype=self.dtype)))

    self._linear_kernel = self.add_variable("linear/%s" % _WEIGHTS_VARIABLE_NAME,
                                            shape=[input_depth + self._num_units, self._num_units * 2])

    self._gate_bias = self.add_variable("gate/%s" % _BIAS_VARIABLE_NAME,
                                        shape=[self._num_units],
                                        initializer=(
                                          self._bias_initializer
                                          if self._bias_initializer is not None
                                          else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    h, n, d, a_max = state

    u = math_ops.matmul(inputs, self._unbounded_kernel)

    linear = math_ops.matmul(array_ops.concat([inputs, h], 1), self._linear_kernel)

    g, a = array_ops.split(value=linear, num_or_size_splits=2, axis=1)

    if not self._normalize:
      u = nn_ops.bias_add(u, self._unbounded_bias)
      g = nn_ops.bias_add(g, self._gate_bias)
    else:
      u = tf.contrib.layers.layer_norm(u)
      g = tf.contrib.layers.layer_norm(g)
      a = tf.contrib.layers.layer_norm(a)

    z = math_ops.multiply(u, math_ops.tanh(g))

    a_newmax = math_ops.maximum(a_max, a)
    exp_diff = math_ops.exp(a_max - a_newmax)
    exp_scaled = math_ops.exp(a - a_newmax)

    n_new = math_ops.multiply(n, exp_diff) + math_ops.multiply(z, exp_scaled)  # Numerically stable update of numerator
    d_new = math_ops.multiply(d, exp_diff) + exp_scaled  # Numerically stable update of denominator
    h_new = self._activation(math_ops.div(n_new, d_new))

    new_state = RWACellTuple(h_new, n_new, d_new, a_newmax)

    return h_new, new_state
