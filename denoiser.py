import tensorflow as tf
from PIL import Image

import numpy as np

import os
import sys
import time

# Should be 55
blur_size = 9

def create_blur_matrix (blur_size):
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
    entity_matrix = np.zeros((blur_size, blur_size, 3, 3))
    for n in range(blur_size):
        for nn in range(blur_size):
            for channels in range(3):
                entity_matrix[n][nn][channels][channels] = 1.0

    blur_matrix = tf.Variable(tf.ones([blur_size * blur_size]),
                                name="blur_matrix")
    blur_matrix = tf.nn.softmax(blur_matrix)
    blur_matrix = tf.reshape(blur_matrix, [9, 9])

    blur_matrix = tf.stack([blur_matrix, blur_matrix, blur_matrix], axis=2)
    blur_matrix = tf.stack([blur_matrix, blur_matrix, blur_matrix], axis=2)
    blur_matrix = blur_matrix * entity_matrix # Masks transformations per channel
    return blur_matrix

def mse(logits, outputs):
    # Calculate cost
    # e = 0.01
    # RelMSE = (y-x)^2 / (x^2 + e)
    # Average over all pixels
    mse = tf.reduce_mean(tf.div(tf.pow(tf.subtract(logits, outputs), 2.0), tf.add(tf.pow(outputs, 2.0), 0.01)))
    return mse


print ("Load images")
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


print ("Create graph")
# Convolute on image
blur_matrix = create_blur_matrix(blur_size)
batch = tf.stack([original_image], name="Packed")
convolved = tf.nn.conv2d(batch, blur_matrix, strides=[1, 1, 1, 1], padding='SAME')
convolved = tf.unstack(convolved, num=1)[0]

mse_cost = mse(target_image, convolved)

train_step = tf.train.AdamOptimizer().minimize(mse_cost)

init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    print ("Run session")
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(100):
        cost = sess.run(train_step)
        if i % 10 == 0:
            print (sess.run(mse_cost))
    result = tf.image.convert_image_dtype(convolved, tf.uint8)
    result_image = Image.fromarray(result.eval(), "RGB")
    result_image.show()

    coord.request_stop()
    coord.join(threads)
