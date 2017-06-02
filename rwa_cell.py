import collections
import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import _RNNCell as RNNCell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.ops import variable_scope as vs
from utils import linear

_checked_scope = core_rnn_cell_impl._checked_scope
RWACellTuple = collections.namedtuple("RWACellTuple", ("h", "n", "d", "a_max"))

class RWACell(RNNCell):
  """Recurrent Weighted Average (cf. http://arxiv.org/abs/1703.01253)."""

  def __init__(self, num_units, input_size=None, activation=tanh, normalize=False, reuse=None):
    if input_size is not None:
      tf.logging.warn("%s: The input_size parameter is deprecated.", self)
    self._num_units = num_units
    self._activation = activation
    self._reuse = reuse
    self._normalize = normalize

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

  def __call__(self, inputs, state, scope=None):
    with _checked_scope(self, scope or "rwa_cell", reuse=self._reuse):
      h, n, d, a_max = state

      with vs.variable_scope("u"):
        u = linear(inputs, self._num_units, True, normalize=self._normalize)

      with vs.variable_scope("g"):
        g = linear([inputs, h], self._num_units, True, normalize=self._normalize)

      with vs.variable_scope("a"): # The bias term when factored out of the numerator and denominator cancels and is unnecessary
        a = linear([inputs, h], self._num_units, False, normalize=self._normalize)

      z = tf.multiply(u, tanh(g))

      a_newmax = tf.maximum(a_max, a)
      exp_diff = tf.exp(a_max - a_newmax)
      exp_scaled = tf.exp(a - a_newmax)

      n_new = tf.multiply(n, exp_diff) + tf.multiply(z, exp_scaled)  # Numerically stable update of numerator
      d_new = tf.multiply(d, exp_diff) + exp_scaled  # Numerically stable update of denominator
      h_new = self._activation(tf.div(n, d))

      new_state = RWACellTuple(h_new, n_new, d_new, a_newmax)

    return h_new, new_state
