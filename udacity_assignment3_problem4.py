# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
#------------------------------------------------------------------------------#
pickle_file = 'notMNIST.pickle'

with open(pickle_file, 'rb') as f:
  save = pickle.load(f)
  train_dataset = save['train_dataset']
  train_labels = save['train_labels']
  valid_dataset = save['valid_dataset']
  valid_labels = save['valid_labels']
  test_dataset = save['test_dataset']
  test_labels = save['test_labels']
  del save  # hint to help gc free up memory
  print('Training set', train_dataset.shape, train_labels.shape)
  print('Validation set', valid_dataset.shape, valid_labels.shape)
  print('Test set', test_dataset.shape, test_labels.shape)
#------------------------------------------------------------------------------#
image_size = 28
num_labels = 10

def reformat(dataset, labels):
  dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
  # Map 1 to [0.0, 1.0, 0.0 ...], 2 to [0.0, 0.0, 1.0 ...]
  labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
  return dataset, labels
train_dataset, train_labels = reformat(train_dataset, train_labels)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
test_dataset, test_labels = reformat(test_dataset, test_labels)
print('Training set', train_dataset.shape, train_labels.shape)
print('Validation set', valid_dataset.shape, valid_labels.shape)
print('Test set', test_dataset.shape, test_labels.shape)
#------------------------------------------------------------------------------#
def accuracy(predictions, labels):
  return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
          / predictions.shape[0])
#------------------------------------------------------------------------------#
import numpy as np

batch_size = 512
hidden_nodes_23 = 1024
hidden_nodes_12 = 64
hidden_nodes_01 = 512
keep_prob = 0.5
init_learning_rate = 0.4
beta = 1e-5

graph = tf.Graph()
with graph.as_default():

  # Input data. For the training data, we use a placeholder that will be fed
  # at run time with a training minibatch.
  tf_train_dataset = tf.placeholder(tf.float32,
                                    shape=(batch_size, image_size * image_size))
  tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
  tf_valid_dataset = tf.constant(valid_dataset)
  tf_test_dataset = tf.constant(test_dataset)
  beta_regul = tf.placeholder(tf.float32)
  
  # Variables for hidden layer 3: put image in
  weights_hidden3 = tf.Variable(
    tf.truncated_normal([image_size * image_size, hidden_nodes_23], stddev=np.sqrt(2.0 / (image_size * image_size))))
  biases_hidden3 = tf.Variable(tf.zeros([hidden_nodes_23]))

  # Variables for hidden layer 2: transform
  weights_hidden2 = tf.Variable(
    tf.truncated_normal([hidden_nodes_23, hidden_nodes_12], stddev=np.sqrt(2.0 / (hidden_nodes_23))))
  biases_hidden2 = tf.Variable(tf.zeros([hidden_nodes_12]))
  
  # Variables for hidden layer 1: transform
  weights_hidden1 = tf.Variable(
    tf.truncated_normal([hidden_nodes_12, hidden_nodes_01], stddev=np.sqrt(2.0 / (hidden_nodes_12))))
  biases_hidden1 = tf.Variable(tf.zeros([hidden_nodes_01]))
  
  # Variables for hidden layer 0: output
  weights = tf.Variable(tf.truncated_normal([hidden_nodes_01, num_labels], stddev=np.sqrt(2.0 / (hidden_nodes_01))))
  biases = tf.Variable(tf.zeros([num_labels]))
    
  # Training computation.
  # relu, L2
  layer3_train = tf.nn.relu(tf.matmul(tf_train_dataset, weights_hidden3) + biases_hidden3)
  drop3 = tf.nn.dropout(layer3_train, keep_prob)
  layer2_train = tf.nn.relu(tf.matmul(drop3, weights_hidden2) + biases_hidden2)
  drop2 = tf.nn.dropout(layer2_train, keep_prob)
  layer1_train = tf.nn.relu(tf.matmul(drop2, weights_hidden1) + biases_hidden1)
  drop1 = tf.nn.dropout(layer1_train, keep_prob)
  logits = tf.matmul(drop1, weights) + biases
  loss = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels) + beta_regul * (tf.nn.l2_loss(weights_hidden3) + \
      tf.nn.l2_loss(weights_hidden2) + tf.nn.l2_loss(weights_hidden1) + tf.nn.l2_loss(weights)))
  
  # Optimizer.
  # GradientDescentOptimizer
  global_step = tf.Variable(0)  # count the number of steps taken.
  learning_rate = tf.train.exponential_decay(init_learning_rate, global_step, 1000, 0.65, staircase=True)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
  
  # Predictions for the training, validation, and test data.
  train_prediction = tf.nn.softmax(logits)  # training data prediction
  # Validation
  layer3_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, weights_hidden3) + biases_hidden3)
  layer2_valid = tf.nn.relu(tf.matmul(layer3_valid, weights_hidden2) + biases_hidden2)
  layer1_valid = tf.nn.relu(tf.matmul(layer2_valid, weights_hidden1) + biases_hidden1)
  valid_prediction = tf.nn.softmax(
    tf.matmul(layer1_valid, weights) + biases)
  # Test
  layer3_test = tf.nn.relu(tf.matmul(tf_test_dataset, weights_hidden3) + biases_hidden3)
  layer2_test = tf.nn.relu(tf.matmul(layer3_test, weights_hidden2) + biases_hidden2)
  layer1_test = tf.nn.relu(tf.matmul(layer2_test, weights_hidden1) + biases_hidden1)
  test_prediction = tf.nn.softmax(tf.matmul(layer1_test, weights) + biases)
#------------------------------------------------------------------------------#
num_steps = 18001

with tf.Session(graph=graph) as session:
  tf.global_variables_initializer().run()
  print("Initialized")
  for step in range(num_steps):
    # Pick an offset within the training data, which has been randomized.
    # Note: we could use better randomization across epochs.
    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
    #offset = np.random.randint(train_dataset.shape[0]-batch_size)
    # Generate a minibatch.
    batch_data = train_dataset[offset:(offset + batch_size), :]
    batch_labels = train_labels[offset:(offset + batch_size), :]
    # Prepare a dictionary telling the session where to feed the minibatch.
    # The key of the dictionary is the placeholder node of the graph to be fed,
    # and the value is the numpy array to feed to it.
    feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, beta_regul : beta}
    _, l, predictions = session.run(
      [optimizer, loss, train_prediction], feed_dict=feed_dict)
    if (step % 500 == 0):
      print("Minibatch loss at step %d: %f" % (step, l))
      print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
      print("Validation accuracy: %.1f%%" % accuracy(
        valid_prediction.eval(), valid_labels))
  print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
