import tensorflow as tf
from PIL import Image

import numpy as np

import os
import sys
import time
import Imath
import OpenEXR

def load_exr(image_path, channels, mergecol=False):
    # Open the input file
    file = OpenEXR.InputFile(image_path)
    # Compute the size
    dw = file.header()['dataWindow']
    sz_x = dw.max.x - dw.min.x + 1
    sz_y = dw.max.y - dw.min.y + 1

    # Read the channels as 32-bit floats
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    image_data = []
    add_channels = ["RenderLayer.DiffCol.R",
                    "RenderLayer.DiffCol.G",
                    "RenderLayer.DiffCol.B",
                    "RenderLayer.SubsurfaceCol.R",
                    "RenderLayer.SubsurfaceCol.G",
                    "RenderLayer.SubsurfaceCol.B",
                    "RenderLayer.GlossCol.R",
                    "RenderLayer.GlossCol.G",
                    "RenderLayer.GlossCol.B",
                    "RenderLayer.TransCol.R",
                    "RenderLayer.TransCol.G",
                    "RenderLayer.TransCol.B",
                    "RenderLayer.Env.R",
                    "RenderLayer.Env.G",
                    "RenderLayer.Env.B"]
    color_channel_r = np.zeros([sz_x*sz_y], dtype = np.float32)
    color_channel_g = np.zeros([sz_x*sz_y], dtype = np.float32)
    color_channel_b = np.zeros([sz_x*sz_y], dtype = np.float32)
    for channel_name in channels:
        numpy_array = np.fromstring( file.channel(channel_name, FLOAT), dtype = np.float32)
        if channel_name == "RenderLayer.Depth.Z":
            numpy_array[numpy_array==10000000000] = 0
            numpy_array = numpy_array + 0.000001
            numpy_array= (numpy_array / (numpy_array.max()*0.5)) - 1.0
        if channel_name in add_channels and mergecol:
            if channel_name.endswith(".R"):
                color_channel_r = color_channel_r + numpy_array
            elif channel_name.endswith(".G"):
                color_channel_g = color_channel_g + numpy_array
            elif channel_name.endswith(".B"):
                color_channel_b = color_channel_b + numpy_array
            continue
        if not channel_name.startswith("RenderLayer.Normal"):
            if mergecol:
                numpy_array[numpy_array>1.0] = 1.0
                numpy_array = (numpy_array*2.0)-1.0
        image_data.append(numpy_array)
    channels_len = len(channels)
    if mergecol:
        image_data.append(color_channel_r)
        image_data.append(color_channel_g)
        image_data.append(color_channel_b)
        channels_len = channels_len-15+3
    image_data = np.array(image_data)
    image_data = image_data.reshape(channels_len, sz_y, sz_x)
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

def create_blur_matrix (blur_size, channel_count):
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
    entity_matrix = np.zeros((blur_size, blur_size, channel_count, channel_count))
    for n in range(blur_size):
        for nn in range(blur_size):
            for channels in range(channel_count):
                entity_matrix[n][nn][channels][channels] = 1.0

    blur_matrix = tf.Variable(tf.random_uniform([blur_size * blur_size], 0.0, 1.0),
                                name="blur_matrix")
    blur_matrix = tf.nn.softmax(blur_matrix)
    blur_matrix = tf.reshape(blur_matrix, [blur_size, blur_size])
    blur_matrix_img = tf.reshape(blur_matrix, [blur_size, blur_size, 1])
    tf.summary.image('blur_matrix', [blur_matrix_img])

    stack = []
    for n in range(channel_count):
        stack.append(blur_matrix)
    blur_matrix = tf.stack(stack, axis=2)
    stack = []
    for n in range(channel_count):
        stack.append(blur_matrix)
    blur_matrix = tf.stack(stack, axis=2)
    blur_matrix = blur_matrix * entity_matrix # Masks transformations per channel
    return blur_matrix


def get_blur_parameters(image):
    #
    kernel_size = 18
    stride_size = 1
    outputs = 27
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
    if 1:
        kernel_size = 9
        stride_size = 1
        outputs = 54
        weight = tf.Variable(tf.random_normal([kernel_size,
                                               kernel_size,
                                               27,
                                               outputs], mean=0, stddev=0.08))
        bias = tf.Variable(tf.random_normal([outputs], mean=0, stddev=0.08))
        x_tensor = tf.nn.conv2d(x_tensor, weight, strides=[1, stride_size, stride_size, 1], padding='SAME')
        x_tensor = tf.nn.bias_add(x_tensor, bias)
        x_tensor = tf.nn.relu6(x_tensor)
    #
    kernel_size = 1
    stride_size = 1
    outputs = 6
    weight = tf.Variable(tf.random_normal([kernel_size,
                                           kernel_size,
                                           54,
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

# SubsurfaceDir.B
# RenderLayer.GlossCol.R
# RenderLayer.Emit.B
# RenderLayer.GlossInd.G
# RenderLayer.SubsurfaceDir.G
# RenderLayer.GlossDir.R
# RenderLayer.TransInd.R
# RenderLayer.GlossDir.G
# RenderLayer.SubsurfaceInd.R
# RenderLayer.SubsurfaceInd.G
# RenderLayer.SubsurfaceInd.B
# RenderLayer.GlossInd.B
# RenderLayer.Shadow.B
# RenderLayer.GlossInd.R
# RenderLayer.SubsurfaceCol.B
# RenderLayer.TransInd.G
# RenderLayer.GlossCol.B
# RenderLayer.SubsurfaceCol.R
# RenderLayer.SubsurfaceCol.G
# RenderLayer.TransInd.B
# RenderLayer.TransCol.R
# RenderLayer.Normal.X
# RenderLayer.Normal.Y
# RenderLayer.Normal.Z
# RenderLayer.SubsurfaceDir.R
# RenderLayer.DiffCol.G
# RenderLayer.DiffCol.R
# RenderLayer.Combined.B
# RenderLayer.DiffDir.R
# RenderLayer.DiffDir.G
# RenderLayer.DiffDir.B
# RenderLayer.Combined.G
# RenderLayer.Env.G
# RenderLayer.GlossDir.B
# RenderLayer.Combined.R
# RenderLayer.TransDir.G
# RenderLayer.Shadow.R
# RenderLayer.DiffInd.B
# RenderLayer.TransCol.G
# RenderLayer.DiffInd.R
# RenderLayer.GlossCol.G
# RenderLayer.Env.B
# RenderLayer.Emit.R
# RenderLayer.Emit.G
# RenderLayer.TransCol.B
# RenderLayer.DiffInd.G
# RenderLayer.DiffCol.B
# RenderLayer.Env.R
# RenderLayer.TransDir.B
# RenderLayer.Combined.A
# RenderLayer.TransDir.R
# RenderLayer.Shadow.G
# RenderLayer.Depth.Z

#low_image = "train_images/box_low.exr"
#high_image = "train_images/box_high.exr"
low_image = "train_images/agent_low.exr"
high_image = "train_images/agent_low.exr"

channels = ["RenderLayer.Combined.R",
            "RenderLayer.Combined.G",
            "RenderLayer.Combined.B",
            "RenderLayer.DiffCol.R",
            "RenderLayer.DiffCol.G",
            "RenderLayer.DiffCol.B",
            "RenderLayer.SubsurfaceCol.R",
            "RenderLayer.SubsurfaceCol.G",
            "RenderLayer.SubsurfaceCol.B",
            "RenderLayer.GlossCol.R",
            "RenderLayer.GlossCol.G",
            "RenderLayer.GlossCol.B",
            "RenderLayer.TransCol.R",
            "RenderLayer.TransCol.G",
            "RenderLayer.TransCol.B",
            "RenderLayer.Normal.X",
            "RenderLayer.Normal.Y",
            "RenderLayer.Normal.Z",
            "RenderLayer.Depth.Z",
            "RenderLayer.Env.R",
            "RenderLayer.Env.G",
            "RenderLayer.Env.B"]
features_image, original_header = load_exr(low_image, channels, mergecol=True)
channels = ["RenderLayer.DiffDir.R",
            "RenderLayer.DiffDir.G",
            "RenderLayer.DiffDir.B",
            "RenderLayer.DiffInd.R",
            "RenderLayer.DiffInd.G",
            "RenderLayer.DiffInd.B",
            "RenderLayer.SubsurfaceDir.R",
            "RenderLayer.SubsurfaceDir.G",
            "RenderLayer.SubsurfaceDir.B",
            "RenderLayer.SubsurfaceInd.R",
            "RenderLayer.SubsurfaceInd.G",
            "RenderLayer.SubsurfaceInd.B",
            "RenderLayer.GlossDir.R",
            "RenderLayer.GlossDir.G",
            "RenderLayer.GlossDir.B",
            "RenderLayer.GlossInd.R",
            "RenderLayer.GlossInd.G",
            "RenderLayer.GlossInd.B",
            "RenderLayer.TransDir.R",
            "RenderLayer.TransDir.G",
            "RenderLayer.TransDir.B",
            "RenderLayer.TransInd.R",
            "RenderLayer.TransInd.G",
            "RenderLayer.TransInd.B"]
original_image, _ = load_exr(low_image, channels)
target_image, _ = load_exr(high_image, channels)

features_image_place = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name="x")
original_image_place = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name="xb")
target_image_place = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name="y")


print ("Create graph")
with tf.name_scope('blur_parameters'):
    # Run get_blur_parameters for:
    # DiffDir
    # DiffInd
    # SubsurfaceDir
    # SubsurfaceInd
    # GlossDir
    # GlossInd
    # TransDir
    # TransInd

    blur_filters_count = 6

    # Analyze image
    blur_parameters = get_blur_parameters(features_image)
    tf.summary.histogram('blur_parameters', blur_parameters)
    blur_parameters = tf.transpose(blur_parameters[0])
    blur_parameters_array = []
    for filter_id in range(blur_filters_count):
        blur_parameters_array.append(tf.transpose(blur_parameters[filter_id]))
        stack = []
        for n in range(len(channels)):
            stack.append(blur_parameters_array[-1])
        blur_parameters_array[-1] = tf.stack(stack, axis=2)
        tf.summary.image('blur_parameters_images_{}'.format(filter_id),
                            [tf.reduce_sum(blur_parameters_array[-1], 2, keep_dims=True)])

with tf.name_scope('apply_blur'):
    blur_matrix_array = []
    final_image = tf.zeros([original_image.shape[0], original_image.shape[1], original_image.shape[2]])
    for filter_id in range(blur_filters_count):
        # Blur image
        blur_matrix_array.append(create_blur_matrix(blur_size, len(channels)))
        blured_image = apply_blur(original_image, blur_matrix_array[-1])
        tf.summary.image('blured_image_{}'.format(filter_id), [tf.reduce_sum(blured_image, 2, keep_dims=True)])
        final_image += (blured_image * blur_parameters_array[filter_id])

# Set cost
with tf.name_scope('train'):
    mse_cost = mse(target_image, final_image)
    tf.summary.scalar('mse_cost', mse_cost)
    train_step = tf.train.AdamOptimizer().minimize(mse_cost)

training = False

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    if not training or True:
        # Restore variables from disk.
        saver.restore(sess, tf.train.latest_checkpoint("/home/gabriel/dev/denoiser/dn-denoiser"))
        print("Model restored.")
    else:
        sess.run(init_op)
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('train', sess.graph)
    print ("Run session")

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for step in range(10000):
        if training:
            summary, cost, parameters, output_image, _ = sess.run([merged, mse_cost, blur_parameters_array[0], final_image, train_step])
        else:
            summary, parameters, output_image, _ = sess.run([merged, blur_parameters_array[0], final_image, train_step])
        if step % 10 == 0 or not training:
            train_writer.add_summary(summary, step)
            save_exr("train_images/output.exr", channels, output_image, original_header)
            if training:
                saver.save(sess, 'denoiser-model', global_step=step)
        if not training:
            break

    coord.request_stop()
    coord.join(threads)
