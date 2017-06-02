import collections
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from utils import linear

_checked_scope = core_rnn_cell_impl._checked_scope
RDACellTuple = collections.namedtuple("RDACellTuple", ("h", "n", "d"))

class RDACell(RNNCell):
  """Recurrent Discounted Attention unit (cf. http://arxiv.org/abs/1705.08480)."""

  def __init__(self, num_units, input_size=None, activation=tanh, normalize=False, reuse=None):
    if input_size is not None:
      tf.logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse
    self._normalize = normalize

  @property
  def state_size(self):
    return RDACellTuple(self._num_units, self._num_units, self._num_units)

  def zero_state(self, batch_size, dtype):
    h, n, d = super(RDACell, self).zero_state(batch_size, dtype)
    return RDACellTuple(h, n, d)

  @property
  def output_size(self):
    return self._num_units

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "rwa_cell", reuse=self._reuse):
      h, n, d = state

      with vs.variable_scope("u"):
        u = linear(inputs, self._num_units, True, normalize=self._normalize)

      with vs.variable_scope("g"):
        g = linear([inputs, h], self._num_units, True, normalize=self._normalize)

      with vs.variable_scope("a"): # The bias term when factored out of the numerator and denominator cancels and is unnecessary
        a = tf.exp(linear([inputs, h], self._num_units, True, normalize=self._normalize))

      with vs.variable_scope("discount_factor"):
        discount_factor = tf.nn.sigmoid(linear([inputs, h], self._num_units, True, normalize=self._normalize))

      z = tf.multiply(u, tanh(g))

      n = tf.multiply(n, discount_factor) + tf.multiply(z, a)  # Numerically stable update of numerator
      d = tf.multiply(d, discount_factor) + a  # Numerically stable update of denominator
      h_new = self._activation(tf.div(n, d))

      new_state = RDACellTuple(h_new, n, d)

    return h_new, new_state
