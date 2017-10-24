import tensorflow as tf
from PIL import Image

import numpy as np

import os
import sys
import time

# Should be 55
blur_size = 9

if blur_size == 3:
    main_matrix = np.array([
        [16, 8, 16],
        [8, 4, 8],
        [16, 8, 16]
    ])
    main_matrix = 1.0 / main_matrix
elif blur_size == 9:
    val = 1.0 / 9.0
    main_matrix = np.array([
        [val, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, val, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, val, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, val, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, val, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, val, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, val, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, val, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, val]
    ])

blur_matrix = tf.Variable(tf.ones([blur_size, blur_size]),
                            name="blur_matrix")

entity_matrix = np.zeros((blur_size, blur_size, 3, 3))
#entity_matrix = tf.zeros([blur_size, blur_size, 3, 3], name="zeros_entity_matrix")
for n in range(blur_size):
    for nn in range(blur_size):
        for rgb_a in range(3):
            entity_matrix[n][nn][rgb_a][rgb_a] = 1.0

print (entity_matrix)

#print (blur_filter.shape)
print ("Create graph")

filepath = 'bl_noise.png'
filename_queue = tf.train.string_input_producer([filepath], capacity=1)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
original_image = tf.image.decode_png(value, name="original", channels=3)
original_image = tf.image.convert_image_dtype(original_image, tf.float32)

filepath = 'bl_clean.png'
filename_queue = tf.train.string_input_producer([filepath], capacity=1)
reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)
target_image = tf.image.decode_png(value, name="target", channels=3)
target_image = tf.image.convert_image_dtype(target_image, tf.float32)


# tf.placeholder(tf.float32, [None, n_classes], name="y")

## Filter

#blur_matrix = tf.Variable(tf.random_normal([blur_size, blur_size,3, 3],
#                                            mean=0, stddev=0.08),
#                            name="blur_matrix")
blur_matrix = tf.Variable(tf.ones([blur_size * blur_size]),
                            name="blur_matrix")

# 1 2 3
# 4 5 6
# 7 8 9

# @

# 1 0 0    2 0 0    3 0 0
# 0 1 0    0 2 0    0 3 0
# 0 0 1    0 0 2    0 0 3

# 4 0 0    5 0 0    6 0 0
# 0 4 0    0 5 0    0 6 0
# 0 0 4    0 0 5    0 0 6

# 7 0 0    8 0 0    9 0 0
# 0 7 0    0 8 0    0 9 0
# 0 0 7    0 0 8    0 0 9

test_filter = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
stacked = np.stack([test_filter, test_filter, test_filter], axis=2)
stacked = np.stack([stacked, stacked, stacked], axis=2)
#mult = test_filter * np.array([[[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]])
#print (stacked)
#lala()


#blur_matrix = tf.reshape(blur_matrix, [81])
blur_matrix = tf.nn.softmax(blur_matrix)
blur_matrix_sm = tf.reshape(blur_matrix, [9, 9])

blur_matrix = tf.stack([blur_matrix_sm, blur_matrix_sm, blur_matrix_sm], axis=2)
blur_matrix = tf.stack([blur_matrix, blur_matrix, blur_matrix], axis=2)
#blur_matrix = tf.reshape(blur_matrix, [9, 9, 3, 3])
#print (blur_matrix)
#lala()

#blur_matrix = tf.nn.relu6(blur_matrix)
#blur_matrix_sm = tf.nn.softmax(blur_matrix, dim=0)
blur_matrix = blur_matrix * entity_matrix # Masks transformations per channel
#blur_matrix = tf.nn.softmax(blur_matrix)
#blur_matrix = tf.tanh(blur_matrix) # -1 to 1
#print (blur_matrix)

# Convolute on image

batch = tf.stack([original_image], name="Packed")
#convolved = tf.nn.conv2d(batch, blur_filter, strides=[1, 1, 1, 1], padding='SAME')
convolved = tf.nn.conv2d(batch, blur_matrix, strides=[1, 1, 1, 1], padding='SAME')
convolved = tf.unstack(convolved, num=1)[0]

# Calculate cost
# e = 0.01
# RelMSE = (y-x)^2 / (x^2 + e)
# Average over all pixels

def mse(logits, outputs):
    mse = tf.reduce_mean(tf.div(tf.pow(tf.subtract(logits, outputs), 2.0), tf.add(tf.pow(outputs, 2.0), 0.01)))
    return mse

mse_cost = mse(target_image, convolved)

train_step = tf.train.AdamOptimizer().minimize(mse_cost)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    print ("Run session")
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10000):
        cost = sess.run(train_step)
        if i % 10 == 0:
            print (sess.run(mse_cost))
            #print (sess.run(blur_matrix))
    result = tf.image.convert_image_dtype(convolved, tf.uint8)
    result_image = Image.fromarray(result.eval(), "RGB")
    result_image.show()

    print (sess.run(blur_matrix))
    print (sess.run(blur_matrix_sm))

    coord.request_stop()
    coord.join(threads)
