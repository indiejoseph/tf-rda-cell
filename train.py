import os
import numpy as np
import tensorflow as tf
import dataplumbing as dp
from tensorflow.contrib import rnn
from tensorflow.contrib.rnn import GRUCell, BasicLSTMCell, LayerNormBasicLSTMCell
from tensorflow.contrib.layers import xavier_initializer as glorot
from rda_cell import RDACell
from rwa_cell import RWACell
from ran_cell import RANCell

flags = tf.app.flags
flags.DEFINE_string("rnn_type", "RDA", "rnn type [RDA, LSTM, GRU]")
FLAGS = flags.FLAGS

def main(_):
  np.random.seed(1)
  tf.set_random_seed(1)
  num_features = dp.train.num_features
  max_steps = dp.train.max_length
  num_cells = 250
  num_classes = dp.train.num_classes
  initialization_factor = 1.0
  num_iterations = 500
  batch_size = 100
  learning_rate = 0.001
  current_step = 0
  initializer = tf.random_uniform_initializer(minval=-np.sqrt(6.0 * 1.0 / (num_cells + num_classes)),
                                              maxval=np.sqrt(6.0 * 1.0 / (num_cells + num_classes)))

  with tf.variable_scope("train", initializer=initializer):
    s = tf.Variable(tf.random_normal([num_cells], stddev=np.sqrt(initialization_factor))) # Determines initial state
    x = tf.placeholder(tf.float32, [batch_size, max_steps, num_features])  # Features
    y = tf.placeholder(tf.float32, [batch_size])  # Labels
    l = tf.placeholder(tf.int32, [batch_size])
    global_step = tf.Variable(0, name="global_step", trainable=False)

    if FLAGS.rnn_type == "RWA":
      cell = RWACell(num_cells)
    elif FLAGS.rnn_type == "RWA_LN":
      cell = RWACell(num_cells, normalize=True)
    elif FLAGS.rnn_type == "RDA":
      cell = RDACell(num_cells)
    elif FLAGS.rnn_type == "RDA_LN":
      cell = RDACell(num_cells, normalize=True)
    elif FLAGS.rnn_type == "RAN":
      cell = RANCell(num_cells)
    elif FLAGS.rnn_type == "RAN_LN":
      cell = RANCell(num_cells, normalize=True)
    elif FLAGS.rnn_type == "LSTM":
      cell = BasicLSTMCell(num_cells)
    elif FLAGS.rnn_type == "LSTM_LN":
      cell = LayerNormBasicLSTMCell(num_cells)
    elif FLAGS.rnn_type == "GRU":
      cell = GRUCell(num_cells)
    else:
      raise Exception('No specified cell')

    states = cell.zero_state(batch_size, tf.float32)

    outputs, states = tf.nn.dynamic_rnn(cell, x, l, states)

    W_o = tf.Variable(tf.random_uniform([num_cells, num_classes],
                                        minval=-np.sqrt(6.0*initialization_factor / (num_cells + num_classes)),
                                        maxval=np.sqrt(6.0*initialization_factor / (num_cells + num_classes))))
    b_o = tf.Variable(tf.zeros([num_classes]))

    if FLAGS.rnn_type == "GRU":
      ly = tf.matmul(states, W_o) + b_o
    else:
      ly = tf.matmul(states.h, W_o) + b_o
    ly_flat = tf.reshape(ly, [batch_size])
    py = tf.nn.sigmoid(ly_flat)

  ##########################################################################################
  # Optimizer/Analyzer
  ##########################################################################################

  # Cost function and optimizer
  #
  cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=ly_flat, labels=y))  # Cross-entropy cost function
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

  # Evaluate performance
  #
  correct = tf.equal(tf.round(py), tf.round(y))
  accuracy = 100.0 * tf.reduce_mean(tf.cast(correct, tf.float32))

  tf.summary.scalar('cost', cost)
  tf.summary.scalar('accuracy', accuracy)

  ##########################################################################################
  # Train
  ##########################################################################################

  # Operation to initialize session
  #
  initializer = tf.global_variables_initializer()
  summaries = tf.summary.merge_all()

  # Open session
  #
  with tf.Session() as session:
    # Summary writer
    #
    summary_writer = tf.summary.FileWriter('log/' + FLAGS.rnn_type, session.graph)

    # Initialize variables
    #
    session.run(initializer)

    # Each training session represents one batch
    #
    for iteration in range(num_iterations):
      # Grab a batch of training data
      #
      xs, ls, ys = dp.train.batch(batch_size)
      feed = {x: xs, l: ls, y: ys}

      # Update parameters
      out = session.run((cost, accuracy, optimizer, summaries, global_step), feed_dict=feed)
      print('Iteration:', iteration, 'Dataset:', 'train', 'Cost:', out[0]/np.log(2.0), 'Accuracy:', out[1])

      summary_writer.add_summary(out[3], current_step)

      # Periodically run model on test data
      if iteration%100 == 0:
        # Grab a batch of test data
        #
        xs, ls, ys = dp.test.batch(batch_size)
        feed = {x: xs, l: ls, y: ys}

        # Run model
        #
        summary_writer.flush()
        out = session.run((cost, accuracy), feed_dict=feed)
        test_cost = out[0] / np.log(2.0)
        test_accuracy = out[1]
        print('Iteration:', iteration, 'Dataset:', 'test', 'Cost:', test_cost, 'Accuracy:', test_accuracy)

      current_step = tf.train.global_step(session, global_step)

    summary_writer.close()

    # Save the trained model
    os.makedirs('bin', exist_ok=True)
    saver = tf.train.Saver()
    saver.save(session, 'bin/train.ckpt')


if __name__ == "__main__":
  tf.app.run()
