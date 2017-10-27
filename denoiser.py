import tensorflow as tf
from PIL import Image

import numpy as np

import os
import sys
import time
import Imath
import OpenEXR

def load_exr(image_path, channels):
    # Open the input file
    file = OpenEXR.InputFile(image_path)
    # Compute the size
    dw = file.header()['dataWindow']
    sz_x = dw.max.x - dw.min.x + 1
    sz_y = dw.max.y - dw.min.y + 1

    # Read the channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    image_data = []
    for channel_name in channels:
        image_data.append(np.fromstring( file.channel(channel_name, FLOAT), dtype = np.float32))
    image_data = np.array(image_data)
    image_data = image_data.reshape(len(channels), sz_y, sz_x)
    image_data = image_data.T
    image_data = np.rot90(image_data, -1)
    return image_data, file.header()

def save_exr(image_path, channel_names, cannels_data, header):
    # Convert to strings
    pixels = {}
    cannels_data = np.rot90(cannels_data, 1)
    to_string = [ Chan.tostring() for Chan in cannels_data.T ]
    channel_id = 0
    for channel_name in channel_names:
        pixels[channel_name] = to_string[channel_id]
        channel_id += 1

    # Write the three color channels to the output file
    out = OpenEXR.OutputFile(image_path, header)
    out.writePixels(pixels)

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
    blur_matrix = tf.reshape(blur_matrix, [blur_size, blur_size])
    blur_matrix_img = tf.reshape(blur_matrix, [blur_size, blur_size, 1])
    tf.summary.image('blur_matrix', [blur_matrix_img])

    blur_matrix = tf.stack([blur_matrix, blur_matrix, blur_matrix], axis=2)
    blur_matrix = tf.stack([blur_matrix, blur_matrix, blur_matrix], axis=2)
    blur_matrix = blur_matrix * entity_matrix # Masks transformations per channel
    return blur_matrix


def get_blur_parameters(image):
    #
    kernel_size = 9
    stride_size = 1
    outputs = 10
    #
    weight = tf.Variable(tf.random_normal([kernel_size,
                                           kernel_size,
                                           image.shape[2],
                                           outputs], mean=0, stddev=0.08))
    bias = tf.Variable(tf.random_normal([outputs], mean=0, stddev=0.08))

    batch = tf.stack([image], name="Packed")
    x_tensor = tf.nn.conv2d(batch, weight, strides=[1, stride_size, stride_size, 1], padding='SAME')
    x_tensor = tf.nn.bias_add(x_tensor, bias)
    x_tensor = tf.nn.relu6(x_tensor)
    #
    kernel_size = 1
    stride_size = 1
    outputs = 3
    weight = tf.Variable(tf.random_normal([kernel_size,
                                           kernel_size,
                                           10,
                                           outputs], mean=0, stddev=0.08))
    bias = tf.Variable(tf.random_normal([outputs], mean=0, stddev=0.08))

    x_tensor = tf.nn.conv2d(x_tensor, weight, strides=[1, stride_size, stride_size, 1], padding='SAME')
    x_tensor = tf.nn.bias_add(x_tensor, bias)
    x_tensor = tf.nn.relu6(x_tensor)

    #x_tensor = tf.reshape(x_tensor, [-1, x_tensor.get_shape().as_list()[1]*x_tensor.get_shape().as_list()[2]*x_tensor.get_shape().as_list()[3]])

    #outputs = 2
    #weights = tf.Variable(tf.random_normal([x_tensor.shape.as_list()[1], outputs], mean=0, stddev=0.08))
    #biases = tf.Variable(tf.random_normal([outputs], mean=0, stddev=0.08))
    #x_tensor = tf.add(tf.matmul(x_tensor, weights), biases)
    #x_tensor = tf.nn.relu6(x_tensor)

    x_tensor = tf.nn.softmax(x_tensor, dim=-1)

    return x_tensor


def apply_blur(image, blur_matrix):
    batch = tf.stack([image], name="Packed")
    convolved = tf.nn.conv2d(batch, blur_matrix, strides=[1, 1, 1, 1], padding='SAME')
    convolved = tf.unstack(convolved, num=1)[0]
    return convolved


def mse(logits, outputs):
    # Calculate cost
    # e = 0.01
    # RelMSE = (y-x)^2 / (x^2 + e)
    # Average over all pixels
    mse = tf.reduce_mean(tf.div(tf.pow(tf.subtract(logits, outputs), 2.0), tf.add(tf.pow(outputs, 2.0), 0.01)))
    return mse


print ("Load images")
# Screen position (x y)
# Color (r g b)
# World position (x y z)
# Shading normal (i j k)
# Texture values for first intersection (r g b)
# Texture values for second intersection (r g b)
# Direct illumination visibility (0 1)

channels = ["RenderLayer.Combined.R", "RenderLayer.Combined.G", "RenderLayer.Combined.B"]

original_image, original_header = load_exr("train_images/box_low.exr", channels)
target_image, _ = load_exr("train_images/box_high.exr", channels)

print ("Create graph")
with tf.name_scope('blur_parameters'):
    # Analyze image
    blur_parameters = get_blur_parameters(original_image)
    tf.summary.histogram('blur_parameters', blur_parameters)
    blur_parameters = tf.transpose(blur_parameters[0])
    blur_parameters_a = tf.transpose(blur_parameters[0])
    blur_parameters_a = tf.stack([blur_parameters_a, blur_parameters_a, blur_parameters_a], axis=2)
    tf.summary.image('blur_parameters_images_a', [blur_parameters_a])
    blur_parameters_b = tf.transpose(blur_parameters[1])
    blur_parameters_b = tf.stack([blur_parameters_b, blur_parameters_b, blur_parameters_b], axis=2)
    tf.summary.image('blur_parameters_images_b', [blur_parameters_b])
    blur_parameters_c = tf.transpose(blur_parameters[2])
    blur_parameters_c = tf.stack([blur_parameters_c, blur_parameters_c, blur_parameters_c], axis=2)
    tf.summary.image('blur_parameters_images_c', [blur_parameters_c])

with tf.name_scope('apply_blur'):
    # Blur image A
    blur_matrix = create_blur_matrix(blur_size)
    blured_image = apply_blur(original_image, blur_matrix)
    tf.summary.image('blured_image', [blured_image])
    # Blur image B
    blur_matrix_b = create_blur_matrix(blur_size)
    blured_image_b = apply_blur(original_image, blur_matrix_b)
    tf.summary.image('blured_image_b', [blured_image_b])
    # Blur image B
    blur_matrix_c = create_blur_matrix(blur_size)
    blured_image_c = apply_blur(original_image, blur_matrix_c)
    tf.summary.image('blured_image_c', [blured_image_c])

    final_image = (blured_image * blur_parameters_a) + (blured_image_b * blur_parameters_b) + (blured_image_c * blur_parameters_c)
    tf.summary.image('final_image', [final_image])

# Set cost
with tf.name_scope('train'):
    mse_cost = mse(target_image, final_image)
    tf.summary.scalar('mse_cost', mse_cost)
    train_step = tf.train.AdamOptimizer().minimize(mse_cost)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    print ("Run session")
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10000):
        summary, cost, parameters, output_image, _ = sess.run([merged, mse_cost, blur_parameters_a, final_image, train_step])
        if i % 10 == 0:
            train_writer.add_summary(summary, i)
            save_exr("train_images/output.exr", channels, output_image, original_header)

    coord.request_stop()
    coord.join(threads)
