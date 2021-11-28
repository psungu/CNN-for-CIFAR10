from tensorflow.compat import v1 as tf
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.utils import shuffle
from statistics import mean
tf.disable_v2_behavior()
import tensorflow as tf2
from scipy import ndimage, misc


def load_pickle(f):

    return  pickle.load(f, encoding='latin1')

def one_hot(a, num_classes):
  return np.squeeze(np.eye(num_classes)[a.reshape(-1)])

def load_batch(filename):

    with open(filename, 'rb') as f:
        datadict = load_pickle(f)
        X = datadict['data']
        Y = datadict['labels']

        X = X.reshape(10000,32,32,3)
        Y =np.array(Y)
        return X, Y

def load_data(ROOT):

    xs = []
    ys = []
    for b in range(1,6):
        f = os.path.join(ROOT, 'data_batch_%d' % (b, ))
        X, Y = load_batch(f)
        xs.append(X)
        ys.append(Y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    del X, Y
    Xte, Yte = load_batch(os.path.join(ROOT, 'test_batch'))
    return Xtr, Ytr, Xte, Yte


def get_data(num_training=49000, num_validation=1000, num_test=10000):
    
    cifar10_dir = './cifar10_data'
    X_train, y_train, X_test, y_test = load_data(cifar10_dir)

    y_train = one_hot(y_train, 10)
    y_test = one_hot(y_test, 10)

    #Define Validation Set
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    X_val= X_val.astype('float32')

    x_train /= 255
    x_test /= 255
    X_val /= 255

    return x_train, y_train, X_val, y_val, x_test, y_test

def augment_training_data(images, labels):

    expanded_images = []
    expanded_labels = []

    j = 0 # counter
    for x, y in zip(images, labels):
        j = j+1
        if j%100==0:
            print ('Expanding data : %03d / %03d' % (j,np.size(images,0)))

        expanded_images.append(x)
        expanded_labels.append(y)

        for i in range(2):
            # rotate the image
            angle = np.random.randint(15, 90)
            new_img = ndimage.rotate(x, angle, reshape=False)

            # shift the image
            shift = np.random.randint(0,1)
            new_img_ = ndimage.shift(new_img, shift)

            expanded_images.append(new_img_)
            expanded_labels.append(y)

    X_train, y_train = shuffle(expanded_images, expanded_labels)


    return np.array(X_train), np.array(y_train)


def model():
    _IMAGE_SIZE = 32
    _IMAGE_CHANNELS = 3
    _NUM_CLASSES = 10

    with tf.name_scope('main_params'):
        x = tf.placeholder(tf.float32, shape=[None, _IMAGE_SIZE * _IMAGE_SIZE * _IMAGE_CHANNELS], name='Input')
        y = tf.placeholder(tf.float32, shape=[None, _NUM_CLASSES], name='Output')
        x_image = tf.reshape(x, [-1, _IMAGE_SIZE, _IMAGE_SIZE, _IMAGE_CHANNELS], name='images')
        is_training = tf.placeholder(tf.bool, name='is_training')
        global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        learning_rate = tf.placeholder(tf.float32, shape=[], name='learning_rate')

    with tf.variable_scope('conv1') as scope:

        conv = tf.layers.conv2d(
            inputs=x_image,
            filters=256,
            kernel_size=[3, 3],
            padding='SAME',
            activation= tf.nn.relu

        )

        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        #batch_norm = tf.compat.v1.layers.batch_normalization(pool, training)
        drop = tf.layers.dropout(pool, rate = 0.35, training=is_training)

    with tf.variable_scope('conv2') as scope:
        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu

        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        drop = tf.layers.dropout(pool, rate=0.35, training=is_training)

    with tf.variable_scope('conv3') as scope:

        conv = tf.layers.conv2d(
            inputs=drop,
            filters=128,
            kernel_size=[3, 3],
            padding='SAME',
            activation=tf.nn.relu
        )
        pool = tf.layers.max_pooling2d(conv, pool_size=[2, 2], strides=2, padding='SAME')
        #batch_norm = tf.compat.v1.layers.batch_normalization(pool, training)
        drop = tf.layers.dropout(pool, rate=0.25, name=scope.name, training=is_training)


    with tf.variable_scope('fully_connected') as scope:
        flat = tf.reshape(drop, [-1, 4 * 4 * 128])

        fc = tf.layers.dense(inputs=flat, units= 512, activation=tf.nn.relu)
        drop = tf.layers.dropout(fc, rate=0.25, name=scope.name, training=is_training)
        softmax = tf.layers.dense(inputs=drop, units=_NUM_CLASSES, name=scope.name)

    y_pred_cls = tf.argmax(softmax, axis=1)

    return x, y, softmax, y_pred_cls, global_step, learning_rate,is_training, fc, flat


def lr(epoch):
    learning_rate = 1e-3

    if epoch > 60:
        learning_rate *= 1e-3
    elif epoch > 40:
        learning_rate *= 1e-2
    elif epoch > 20:
        learning_rate *= 1e-1
    return learning_rate