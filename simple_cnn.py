import tensorflow as tf
import numpy as np
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn

print("Loading data...")
x_text, y = data_helpers.load_data_and_labels("./data/rt-polaritydata/rt-polarity.pos", "./data/rt-polaritydata/rt-polarity.neg")

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))

# Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

# Split train/test set
dev_sample_index = -1 * int(.1 * float(len(y)))
x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

del x, y, x_shuffled, y_shuffled

print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


######################## TensorFlow Graph 구성

# 값 초기화
sequence_length = x_train.shape[1]
num_classes = y_train.shape[1]
vocab_size = len(vocab_processor.vocabulary_)
embedding_size = 128
filter_sizes = [3,4,5]
num_filters = 50
dropout_keep_prob = 0.5
num_epoch = 50
batch_size = 100

# X, Y 정의
input_x = tf.placeholder(tf.int32, [None, sequence_length], name='input_x')
input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

# Embedding Layer
W = tf.Variable(tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
embedded_chars = tf.nn.embedding_lookup(W, input_x)
embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

# Convolution and Max-Pooling
pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):
    filter_shape = [filter_size, embedding_size, 1, num_filters]
    W_c = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W_c")
    conv = tf.nn.conv2d(
        embedded_chars_expanded,
        W_c,
        strides=[1, 1, 1, 1],
        padding="VALID",
        name="conv")
    h = tf.nn.relu(conv, name="relu_c")
    pooled = tf.nn.max_pool(
        h,
        ksize=[1, sequence_length - filter_size + 1, 1, 1],
        strides=[1, 1, 1, 1],
        padding='VALID',
        name="pool")
    pooled_outputs.append(pooled)

# Flatten and Dropout
num_filters_total = num_filters * len(filter_sizes)
h_pool = tf.concat(pooled_outputs, 3)
h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
h_drop = tf.nn.dropout(h_pool_flat, dropout_keep_prob)

# Final Fully Connected Layer
W_f = tf.get_variable(
                "W_f",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
b_f = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b_f")
scores = tf.nn.xw_plus_b(h_drop, W_f, b_f, name="scores")
predictions = tf.argmax(scores, 1, name="predictions")

#loss
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=input_y))

#accuracy
correct_predictions = tf.equal(predictions, tf.argmax(input_y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))

#optimizer
optimizer = tf.train.AdamOptimizer(1e-3).minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_len = len(x_train)
for epoch in range(num_epoch):
    avg_loss = 0
    avg_acc = 0
    total_batch = int(total_len/batch_size)
    for i in range(total_batch):
        st = i*batch_size
        en = min((i+1)*batch_size,total_len)

        batch_xs = x_train[st:en]
        batch_ys = y_train[st:en]

        feed_dict = {input_x:batch_xs, input_y:batch_ys}

        acc, l, _ = sess.run([accuracy,loss, optimizer], feed_dict=feed_dict)

        avg_loss += l / total_batch
        avg_acc +=  acc / total_batch

    print('Epoch:', '%03d' % (epoch + 1), 'loss =', '{:.6f}'.format(avg_loss), 'accuracy =', '{:.6f}'.format(avg_acc))

feed_dict = {input_x:x_dev, input_y:y_dev}

acc = sess.run(accuracy, feed_dict=feed_dict)
print ('Test Accuraacy : ', acc)