import collections
import tensorflow as tf
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import array_ops, math_ops, init_ops, nn_ops

RANCellTuple = collections.namedtuple("RANCellTuple", ("c", "h"))
_WEIGHTS_VARIABLE_NAME = rnn_cell_impl._WEIGHTS_VARIABLE_NAME
_BIAS_VARIABLE_NAME = rnn_cell_impl._BIAS_VARIABLE_NAME

class RANCell(rnn_cell_impl._LayerRNNCell):
  """Recurrent Additive Networks (cf. https://arxiv.org/abs/1705.07393)."""

  def __init__(self, num_units, activation=math_ops.tanh, normalize=False,
               bias_initializer=None, reuse=None, name=None):
    super(RANCell, self).__init__(_reuse=reuse, name=name)

    self._num_units = num_units
    self._activation = activation
    self._normalize = normalize
    self._reuse = reuse
    self._bias_initializer = bias_initializer

  @property
  def state_size(self):
    return RANCellTuple(self._num_units, self._num_units)

  def zero_state(self, batch_size, dtype):
    c, h = super(RANCell, self).zero_state(batch_size, dtype)
    return RANCellTuple(c, h)

  @property
  def output_size(self):
    return self._num_units

  def build(self, inputs_shape):
    if inputs_shape[1].value is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[1].value

    self._context_kernel = self.add_variable("context/%s" % _WEIGHTS_VARIABLE_NAME,
                                               shape=[input_depth, self._num_units])

    self._context_bias = self.add_variable("context/%s" % _BIAS_VARIABLE_NAME,
                                             shape=[self._num_units],
                                             initializer=(
                                                self._bias_initializer
                                                if self._bias_initializer is not None
                                                else init_ops.zeros_initializer(dtype=self.dtype)))

    self._linear_kernel = self.add_variable("linear/%s" % _WEIGHTS_VARIABLE_NAME,
                                            shape=[input_depth + self._num_units, self._num_units * 2])

    self._gates_bias = self.add_variable("gates/%s" % _BIAS_VARIABLE_NAME,
                                         shape=[self._num_units],
                                         initializer=(
                                           self._bias_initializer
                                           if self._bias_initializer is not None
                                           else init_ops.zeros_initializer(dtype=self.dtype)))

    self.built = True

  def call(self, inputs, state):
    c, h = state

    content = math_ops.matmul(inputs, self._context_kernel)

    linear = math_ops.matmul(array_ops.concat([inputs, h], 1), self._linear_kernel)

    if not self._normalize:
      content = nn_ops.bias_add(content, self._context_bias)
      linear = nn_ops.bias_add(linear, self._gates_bias)
    else:
      content = tf.contrib.layers.layer_norm(content)
      linear = tf.contrib.layers.layer_norm(linear)

    gates = tf.nn.sigmoid(linear)
    i, f = array_ops.split(value=gates, num_or_size_splits=2, axis=1)
    new_c = i * content + f * c
    new_h = self._activation(c)
    new_state = RANCellTuple(new_c, new_h)
    output = new_h
    return output, new_state
