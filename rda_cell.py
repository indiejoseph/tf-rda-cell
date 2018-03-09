import collections
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops, math_ops, init_ops, nn_ops

RDACellTuple = collections.namedtuple("RDACellTuple", ("h", "n", "d"))
_WEIGHTS_VARIABLE_NAME = rnn_cell_impl._WEIGHTS_VARIABLE_NAME
_BIAS_VARIABLE_NAME = rnn_cell_impl._BIAS_VARIABLE_NAME

class RDACell(rnn_cell_impl._LayerRNNCell):
  """Recurrent Discounted Attention unit (cf. http://arxiv.org/abs/1705.08480)."""

  def __init__(self, num_units, activation=math_ops.tanh, normalize=False,
               bias_initializer=None, reuse=None, name=None):
    super(RDACell, self).__init__(_reuse=reuse, name=name)

    self._num_units = num_units
    self._activation = activation
    self._normalize = normalize
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return RDACellTuple(self._num_units, self._num_units, self._num_units)

  def zero_state(self, batch_size, dtype):
    h, n, d = super(RDACell, self).zero_state(batch_size, dtype)
    return RDACellTuple(h, n, d)

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
                                            shape=[input_depth + self._num_units, self._num_units * 3])

    self._gate_bias = self.add_variable("gate/%s" % _BIAS_VARIABLE_NAME,
                                        shape=[self._num_units],
                                        initializer=(
                                          self._bias_initializer
                                          if self._bias_initializer is not None
                                          else init_ops.zeros_initializer(dtype=self.dtype)))

    self._discount_bias = self.add_variable("discount_factor/%s" % _BIAS_VARIABLE_NAME,
                                            shape=[self._num_units],
                                            initializer=(
                                               self._bias_initializer
                                               if self._bias_initializer is not None
                                               else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    h, n, d = state

    u = math_ops.matmul(inputs, self._unbounded_kernel)

    linear = math_ops.matmul(array_ops.concat([inputs, h], 1), self._linear_kernel)

    g, a, discount_factor = array_ops.split(value=linear, num_or_size_splits=3, axis=1)

    if not self._normalize:
      u = nn_ops.bias_add(u, self._unbounded_bias)
      g = nn_ops.bias_add(g, self._gate_bias)
      discount_factor = nn_ops.bias_add(discount_factor, self._discount_bias)
    else:
      u = tf.contrib.layers.layer_norm(u)
      g = tf.contrib.layers.layer_norm(g)
      a = tf.contrib.layers.layer_norm(a)
      discount_factor = tf.contrib.layers.layer_norm(discount_factor)

    a = math_ops.exp(a)
    discount_factor = tf.nn.sigmoid(discount_factor)
    z = math_ops.multiply(u, math_ops.tanh(g))
    n = math_ops.multiply(n, discount_factor) + math_ops.multiply(z, a) # Numerically stable update of numerator
    d = math_ops.multiply(d, discount_factor) + a # Numerically stable update of denominator
    h_new = self._activation(math_ops.div(n, d))

    new_state = RDACellTuple(h_new, n, d)

    return h_new, new_state
