import tensorflow as tf
from PIL import Image

import numpy as np

import os
import sys
import time

print ("Convolve an image with a 3x3 blur filter")

filename_queue = tf.train.string_input_producer(['044761b99582589dfccb127507e5d962.jpg'], capacity=1)

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

original_image = tf.image.decode_jpeg(value, name="avatar", channels=3)
original_image = tf.image.convert_image_dtype(original_image, tf.float32)

one_sixteenth = 1.0 / 16
one_eighth = 1.0 / 8
one_quarter = 1.0 / 4

filter_row_1 = [
    [[ one_sixteenth, 0, 0],
     [0, one_sixteenth, 0],
     [0, 0, one_sixteenth]],
    [[ one_eighth, 0, 0],
     [0, one_eighth, 0],
     [0, 0, one_eighth]],
    [[ one_sixteenth, 0, 0],
     [0, one_sixteenth, 0],
     [0, 0, one_sixteenth]]
]

filter_row_2 = [
    [[ one_eighth, 0, 0],
     [0, one_eighth, 0],
     [0, 0, one_eighth]],
    [[ one_quarter, 0, 0],
     [0, one_quarter, 0],
     [0, 0, one_quarter]],
    [[ one_eighth, 0, 0],
     [0, one_eighth, 0],
     [0, 0, one_eighth]]
]

filter_row_3 = [
    [[ one_sixteenth, 0, 0],
     [0, one_sixteenth, 0],
     [0, 0, one_sixteenth]],
    [[ one_eighth, 0, 0],
     [0, one_eighth, 0],
     [0, 0, one_eighth]],
    [[ one_sixteenth, 0, 0],
     [0, one_sixteenth, 0],
     [0, 0, one_sixteenth]]
]

blur_filter = [filter_row_1, filter_row_2, filter_row_3]

init_op = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(1):
        batch = tf.stack([original_image], name="Packed")

        convolved = tf.nn.conv2d(batch, blur_filter, strides=[1, 1, 1, 1], padding='VALID')
        convolved = tf.unstack(convolved, num=1)[0]

        result = tf.image.convert_image_dtype(convolved, tf.uint8)

        result_image = Image.fromarray(result.eval(), "RGB")

        result_image.show()

    coord.request_stop()
    coord.join(threads)
