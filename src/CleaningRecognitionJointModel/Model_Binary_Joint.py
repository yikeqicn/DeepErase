# Unet
from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf

import cv2

import os, sys
import numpy as np
import math
from datetime import datetime
import time
from PIL import Image
from math import ceil
from collections import OrderedDict
import logging
from utils_seg import get_image_summary, log_images, _variable_with_weight_decay, _variable_on_cpu, _add_loss_summaries, \
    _activation_summary, print_hist_summery, get_hist, per_class_acc, writeImage

# Recognition
from glob import glob
import numpy as np
import sys
import tensorflow as tf
from os.path import join
from Densenet4htr import Densenet4htr
import utils_recg  # dangerous


# model layers
def weight_variable(shape, stddev=0.1, name="weight"):
    shape = np.array(shape)
    # print(shape)
    # print(stddev)
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def weight_variable_devonc(shape, stddev=0.1, name="weight_devonc"):
    shape = np.array(shape)
    return tf.Variable(tf.truncated_normal(shape, stddev=stddev), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv2d(x, W, b, keep_prob_):
    with tf.name_scope("conv2d"):
        conv_2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # 'VALID'
        conv_2d_b = tf.nn.bias_add(conv_2d, b)
        return tf.nn.dropout(conv_2d_b, keep_prob_)


def deconv2d(x, W, stride):
    with tf.name_scope("deconv2d"):
        x_shape = tf.shape(x)
        output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] // 2])
        return tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, stride, stride, 1], padding='SAME',
                                      name="conv2d_transpose")  # 'VALID'


def max_pool(x, n):
    return tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')  # 'VALID'


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], 3)


def pixel_wise_softmax(output_map):
    with tf.name_scope("pixel_wise_softmax"):
        max_axis = tf.reduce_max(output_map, axis=3, keepdims=True)
        exponential_map = tf.exp(output_map - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize


def cross_entropy(y_, output_map):
    return -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(output_map, 1e-10, 1.0)), name="cross_entropy")


# unet setting
def create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2,
                    summaries=True):
    """
    Creates a new convolutional unet for the given parametrization.
    :param x: input tensor, shape [?,nx,ny,channels]
    :param keep_prob: dropout probability tensor
    :param channels: number of channels in the input image
    :param n_class: number of output labels
    :param layers: number of layers in the net
    :param features_root: number of features in the first layer
    :param filter_size: size of the convolution filter
    :param pool_size: size of the max pooling operation
    :param summaries: Flag if summaries should be created
    """

    logging.info(
        "Layers {layers}, features {features}, filter size {filter_size}x{filter_size}, pool size: {pool_size}x{pool_size}".format(
            layers=layers,
            features=features_root,
            filter_size=filter_size,
            pool_size=pool_size))

    # Placeholder for the input image
    with tf.name_scope("preprocessing"):
        nx = tf.shape(x)[1]
        ny = tf.shape(x)[2]
        # nx=32
        # ny=128
        # channels=1
        x_image = tf.reshape(x, tf.stack([-1, nx, ny, channels]))
        in_node = x_image
        batch_size = tf.shape(x_image)[0]

    weights = []
    biases = []
    convs = []
    pools = OrderedDict()
    deconv = OrderedDict()
    dw_h_convs = OrderedDict()
    up_h_convs = OrderedDict()

    in_size = 1000  # ?????????????????????
    size = in_size
    # down layers
    for layer in range(0, layers):
        with tf.name_scope("down_conv_{}".format(str(layer))):
            features = 2 ** layer * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))
            if layer == 0:
                w1 = weight_variable([filter_size, filter_size, channels, features], stddev, name="w1")
            else:
                w1 = weight_variable([filter_size, filter_size, features // 2, features], stddev, name="w1")

            w2 = weight_variable([filter_size, filter_size, features, features], stddev, name="w2")
            b1 = bias_variable([features], name="b1")
            b2 = bias_variable([features], name="b2")

            conv1 = conv2d(in_node, w1, b1, keep_prob)
            print(str(layer) + ' conv1: ' + str(conv1.get_shape()))
            tmp_h_conv = tf.nn.relu(conv1)
            conv2 = conv2d(tmp_h_conv, w2, b2, keep_prob)
            print(str(layer) + ' conv2: ' + str(conv2.get_shape()))
            dw_h_convs[layer] = tf.nn.relu(conv2)

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size -= 2 * 2 * (filter_size // 2)  # valid conv
            if layer < layers - 1:
                pools[layer] = max_pool(dw_h_convs[layer], pool_size)
                in_node = pools[layer]
                size /= pool_size

    in_node = dw_h_convs[layers - 1]

    # up layers
    for layer in range(layers - 2, -1, -1):
        with tf.name_scope("up_conv_{}".format(str(layer))):
            features = 2 ** (layer + 1) * features_root
            stddev = np.sqrt(2 / (filter_size ** 2 * features))

            wd = weight_variable_devonc([pool_size, pool_size, features // 2, features], stddev, name="wd")
            bd = bias_variable([features // 2], name="bd")
            h_deconv = tf.nn.relu(deconv2d(in_node, wd, pool_size) + bd)
            print(str(layer) + ' h_deconv: ' + str(h_deconv.get_shape()))
            h_deconv_concat = crop_and_concat(dw_h_convs[layer], h_deconv)
            print(str(layer) + ' h_deconv_concat: ' + str(h_deconv_concat.get_shape()))
            deconv[layer] = h_deconv_concat

            w1 = weight_variable([filter_size, filter_size, features, features // 2], stddev, name="w1")
            w2 = weight_variable([filter_size, filter_size, features // 2, features // 2], stddev, name="w2")
            b1 = bias_variable([features // 2], name="b1")
            b2 = bias_variable([features // 2], name="b2")

            conv1 = conv2d(h_deconv_concat, w1, b1, keep_prob)
            h_conv = tf.nn.relu(conv1)
            print(str(layer) + ' h_conv1_post_deconv: ' + str(h_conv.get_shape()))
            conv2 = conv2d(h_conv, w2, b2, keep_prob)
            in_node = tf.nn.relu(conv2)
            up_h_convs[layer] = in_node
            print(str(layer) + ' h_conv2_post_deconv: ' + str(in_node.get_shape()))

            weights.append((w1, w2))
            biases.append((b1, b2))
            convs.append((conv1, conv2))

            size *= pool_size
            size -= 2 * 2 * (filter_size // 2)  # valid conv

    # Output Map
    with tf.name_scope("output_map"):
        weight = weight_variable([1, 1, features_root, n_class], stddev)
        bias = bias_variable([n_class], name="bias")
        conv = conv2d(in_node, weight, bias, tf.constant(1.0))
        print(str(layer) + ' outmap: ' + str(conv.get_shape()))

        # output_map = tf.nn.relu(conv)
        output_map = conv  # no activation, to be consistant with other models and leverage previous loss/prediction structures yike !!!!
        up_h_convs["out"] = output_map

    if summaries:
        with tf.name_scope("summaries"):
            for i, (c1, c2) in enumerate(convs):
                tf.summary.image('summary_conv_%02d_01' % i, get_image_summary(c1))
                tf.summary.image('summary_conv_%02d_02' % i, get_image_summary(c2))

            for k in pools.keys():
                tf.summary.image('summary_pool_%02d' % k, get_image_summary(pools[k]))

            for k in deconv.keys():
                tf.summary.image('summary_deconv_concat_%02d' % k, get_image_summary(deconv[k]))

            for k in dw_h_convs.keys():
                tf.summary.histogram("dw_convolution_%02d" % k + '/activations', dw_h_convs[k])

            for k in up_h_convs.keys():
                tf.summary.histogram("up_convolution_%s" % k + '/activations', up_h_convs[k])

    variables = []
    for w1, w2 in weights:
        variables.append(w1)
        variables.append(w2)

    for b1, b2 in biases:
        variables.append(b1)
        variables.append(b2)

    return output_map, variables, int(in_size - size)


class DecoderType:
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model:
    # model constants
    # batchSize = 50 #qyk
    # imgSize = (128, 32)
    # imgSize = (192, 48) #qyk
    maxTextLen = 32  # qyk?
    MOVING_AVERAGE_DECAY = 0.9999  # The decay to use for the moving average.
    NUM_EPOCHS_PER_DECAY = 350.0  # Epochs after which learning rate decays.
    LEARNING_RATE_DECAY_FACTOR = 0.1  # Learning rate decay factor.

    def __init__(self, args, charList, loss_beta, loss_weight, decoderType=DecoderType.BestPath, experiment=None,
                 mustRestore_seg=False, mustRestore_recg=False, joint=False):  # !!!!!!!!!!!!!!!!!!!!!!!!
        '''
        loss_betaxsegloss+(1-loss_beta)xrecgloss
        loss_weight: used in segnet training
        joint: False -> train recognition only, True -> train segmentation with recognition frozen
        '''
        self.loss_beta = loss_beta  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.args = args
        self.experiment = experiment
        self.lrInit = args.lrInit
        # self.mustRestore_recg= mustRestore_recg
        ###################################
        "init segnet model parameters:"
        ###################################
        self.mustRestore_seg = mustRestore_seg
        ###model hyperparameters###
        self.num_classes = args.num_class
        # self.FilePaths = FilePaths
        self.batch_size_seg = args.batch_size_seg  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.loss_weight = loss_weight

        ###########################################################################
        "init recognition model parameters: add CNN, RNN and CTC and initialize TF"
        ###########################################################################
        self.charList = charList
        self.decoderType = decoderType
        self.mustRestore_recg = mustRestore_recg
        # self.FilePaths = FilePaths
        self.batchsize_recg = args.batchsize_recg  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # self.lrInit = args.lrInit #!!!!!!!!!!!!!!!!!!!!!!!!
        # self.args = args #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        ############################################################################
        """Graph Set Up"""
        ############################################################################
        tf.reset_default_graph()  # yike reset default graph  #!!!!!!!!!!!!!!!!!!!!!还要吗？????????
        with tf.name_scope('graph_segmentation'):
            # self.loss_segmentation, output = YIKE_FUNCTION_HERE()

            ###input### -- try to only set up graph once, combine train and test, by yike
            # tf.reset_default_graph() # yike reset default graph
            self.input_images_seg = tf.placeholder(tf.float32, shape=[None, self.args.image_h, self.args.image_w,
                                                                      self.args.image_c])  # try my best to make runtime batch_size flexible #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.input_labels_seg = tf.placeholder(tf.int64, shape=[None, self.args.image_h, self.args.image_w,
                                                                    1])  # !!!!!!!!!!!!!!!!!!!!!
            self.phase_train = tf.placeholder(tf.bool, name='phase_train')

            ###graph### -- combine
            self.logit_seg = self.setup_graph(self.input_images_seg,
                                              self.phase_train)  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            self.loss_seg = self.cal_loss(self.logit_seg, self.input_labels_seg)*500 # make it to same level as recg loss
            self.pred_seg = tf.argmax(self.logit_seg, axis=3)

            input_images_2d_seg = tf.squeeze(self.input_images_seg, [3])  # to 2d images, channel=1
            self.output_clean_seg = tf.to_float(self.pred_seg) * (255 - input_images_2d_seg) + input_images_2d_seg

            print('clean output from seg: ' + str(self.output_clean_seg.get_shape()))

        with tf.name_scope('graph_recognition'):
            # assume the input has been resized to 32x128
            if joint:
                self.input_images_recg = tf.transpose(self.output_clean_seg, perm=(0, 2, 1))
            else:
                self.input_images_recg = tf.placeholder(tf.float32, shape=(None, args.image_w, args.image_h))
            #self.input_images_recg = tf.transpose(self.output_clean_seg, perm=(0, 2, 1))
            #self.input_images_recg = tf.placeholder(tf.float32, shape=(None, args.image_w, args.image_h))
            print('recg input: ' + str(self.input_images_recg.get_shape()))
            # CNN
            if args.nondensenet:
                cnnOut4d = self.setupCNN(self.input_images_recg)
            else:  # use densenet by default
                cnnOut4d = self.setupCNNdensenet(self.input_images_recg, args)

            # RNN
            rnnOut3d = self.setupRNN(cnnOut4d)

            # CTC
            (self.ctcloss, self.decoder) = self.setupCTC(rnnOut3d)

            # Explicit regularizers
            self.loss_recg = self.ctcloss + args.wdec * self.setupWdec(args)

        # combine losses
        self.loss_total = (1 - loss_beta) * self.loss_recg + loss_beta * self.loss_seg
        print(self.loss_total)
        # optimizer for NN parameters
        self.batchesTrained = args.batchesTrained  # only for recognition training
        self.learning_rate = tf.placeholder(tf.float32, shape=[])  # for recognition and segmentation
        self.global_step = tf.Variable(0, trainable=False)  # for segmentation training

        self.var_list_train_seg = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "graph_segmentation") #TRAINABLE_
        self.obj_list_savable_seg= tf.get_collection(tf.GraphKeys.VARIABLES, "graph_segmentation")
        self.train_op_seg = self.train_op_seg_prepare(total_loss=self.loss_total, lr=self.learning_rate,
                                                      global_step=self.global_step, var_list=self.var_list_train_seg)
        # self.learning_rate self.loss_total
        ## optimizer for recognition only

        self.var_list_train_recg = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "graph_recognition")#TRAINABLE_
        self.obj_list_savable_recg= tf.get_collection(tf.GraphKeys.VARIABLES, "graph_recognition")
        if args.optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss_recg,
                                                                                    var_list=self.var_list_train_recg)
        elif args.optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss_recg,
                                                                                 var_list=self.var_list_train_recg)
        elif args.optimizer == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, .9).minimize(self.loss_recg,
                                                                                         var_list=self.var_list_train_recg)

        # self.global_step,var_list=self.var_list_train_seg)  # !!!!!!!!!!!!!!!!!!!!!!!!!!
        # above: loss need to change to total loss
        ###session and saver###
        (self.sess, self.saver_seg, self.saver_recg) = self.initTF()  # tobe changed!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    ############################################################
    #####               Segnet Functions             ###########
    ############################################################ Not Adjusted !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    ### 1. loss factory ###

    def weighted_loss(self, logits, labels):  # num_classes, head=None):
        """ median-frequency re-weighting """
        with tf.name_scope('loss'):
            # print('w_llll')
            logits = tf.reshape(logits, (-1, self.num_classes))
            # print(logits.get_shape())
            epsilon = tf.constant(value=1e-10)

            logits = logits + epsilon

            # consturct one-hot label array
            label_flat = tf.reshape(labels, (-1, 1))
            # print(label_flat.get_shape())

            # should be [batch ,num_classes]
            labels = tf.reshape(tf.one_hot(label_flat, depth=self.num_classes), (-1, self.num_classes))
            # print(labels.get_shape())

            softmax = tf.nn.softmax(logits)
            # print(softmax.get_shape())
            #        print(epsilon.get_shape())

            #        print((labels * tf.log(softmax + epsilon)).get_shape())
            #        print(head.shape)
            #        print(tf.multiply(labels * tf.log(softmax + epsilon), head))

            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax + epsilon), self.loss_weight),
                                           axis=[1])
            #        print(cross_entropy.get_shape()) # yike head -> self.loss_weight

            cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
            #        print(cross_entropy_mean.get_shape())
            tf.add_to_collection('losses', cross_entropy_mean)

            loss = tf.add_n(tf.get_collection('losses'), name='total_loss_seg')
            print('loss_seg: ' + str(loss.get_shape()))

        return loss

    def cal_loss(self, logits, labels):
        labels = tf.cast(labels, tf.int32)
        return self.weighted_loss(logits, labels)

        # self.weighted_loss(logits, labels, num_classes=NUM_CLASSES, head=loss_weight)

        ###2. train optimizer factory ##

    def train_op_seg_prepare(self, total_loss, lr, global_step, var_list):
        # all of them are tensor
        # total_sample = 274 yike: ok to comment out?
        # num_batches_per_epoch = 274/1 yike: ok to comment out?

        loss_averages_op = _add_loss_summaries(total_loss)
        # Compute gradients.
        with tf.control_dependencies([loss_averages_op]):
            # print('try...')
            opt = tf.train.AdamOptimizer(lr)
            print('toto_loss_shape: ' + str(total_loss))
            opt.compute_gradients(total_loss, var_list=var_list)  # add list of variables
            grads = opt.compute_gradients(total_loss, var_list=var_list)  # !!!!!!!
            # print(grads)
            apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

            # Add histograms for trainable variables.
            #######for var in tf.trainable_variables():
            ######tf.summary.histogram(var.op.name, var)

            # Add histograms for gradients.
            #####for grad, var in grads:
            #####if grad is not None:
            #####tf.summary.histogram(var.op.name + '/gradients', grad)

            # Track the moving averages of all trainable variables.
            variable_averages = tf.train.ExponentialMovingAverage(Model.MOVING_AVERAGE_DECAY, global_step)
            variables_averages_op = variable_averages.apply(var_list=var_list)  # tf.trainable_variables()

            with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                train_op_seg = tf.no_op(name='train_op_seg_prepare')

        return train_op_seg

        ###3. graph factory ###

    def setup_graph(self, images, phase_train):
        # previous inference() labels,inference, batch_size -- in order to get batch_size at running time
        # rather than using fixed batch_size in graph set up, revise it in inference:
        # batchsize=tf.shape(images)[0] # yike !!!
        print('GGG')
        input_shape = images.get_shape().as_list()
        print(input_shape)

        #       create_conv_net(x, keep_prob, channels, n_class, layers=3, features_root=16, filter_size=3, pool_size=2,
        #                    summaries=True)

        logit, _, __ = create_conv_net(x=images, keep_prob=0.8, channels=input_shape[3], n_class=self.num_classes,
                                       layers=3, features_root=32, filter_size=3)
        print(logit.get_shape())
        """
         Start Classify 

        # output predicted class number (6)
        with tf.variable_scope('conv_classifier') as scope:
          kernel = _variable_with_weight_decay('weights',
                                            shape=[1, 1, 64, self.num_classes],
                                            initializer=msra_initializer(1, 64),
                                            wd=0.0005)
          conv = tf.nn.conv2d(conv_decode1, kernel, [1, 1, 1, 1], padding='SAME')
          print('cv')
          print(conv.get_shape())
          biases = _variable_on_cpu('biases', [self.num_classes], tf.constant_initializer(0.0))
          print(biases.get_shape())
          logit= tf.nn.bias_add(conv, biases, name=scope.name)
          #conv_classifier = tf.nn.bias_add(conv, biases, name=scope.name)
          #print(conv_classifier.get_shape())
          #logit = conv_classifier
          #print('LLL')
          #print(labels)
          #print(conv_classifier)

          #loss = cal_loss(conv_classifier, labels)
          print(logit.get_shape())
          """
        return logit  # loss

    ############################################################################
    ###                 Recognition Functions                                ###
    ############################################################################
    def setupCNN(self, cnnIn3d):
        "vanilla cnn from original github repo"
        cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)

        # list of parameters for the layers
        kernelVals = [5, 5, 3, 3, 3]
        featureVals = [1, 32, 64, 128, 128, 256]
        strideVals = poolVals = [(2, 2), (2, 2), (1, 2), (1, 2), (1, 2)]
        numLayers = len(strideVals)

        # create layers
        pool = cnnIn4d  # input to first CNN layer
        for i in range(numLayers):
            kernel = tf.Variable(
                tf.truncated_normal([kernelVals[i], kernelVals[i], featureVals[i], featureVals[i + 1]], stddev=0.1))
            conv = tf.nn.conv2d(pool, kernel, padding='SAME', strides=(1, 1, 1, 1))
            relu = tf.nn.relu(conv)
            pool = tf.nn.max_pool(relu, (1, poolVals[i][0], poolVals[i][1], 1),
                                  (1, strideVals[i][0], strideVals[i][1], 1),
                                  'VALID')

        self.is_training = tf.placeholder(tf.bool, shape=[])  # dummy placeholder to prevent error, no effect
        return pool

    def setupCNNdensenet(self, cnnIn3d, args):
        "ADDED BY RONNY: densenet cnn"
        print('shape of cnn input: ' + str(cnnIn3d.get_shape().as_list()))
        cnnIn4d = tf.expand_dims(input=cnnIn3d, axis=3)
        net = Densenet4htr(cnnIn4d, **vars(args))
        self.is_training = net.is_training
        print('shape of cnn output: ' + str(net.output.get_shape().as_list()))
        return net.output

    def setupRNN(self, rnnIn4d):
        "create RNN layers and return output of these layers"
        rnnIn3d = tf.squeeze(rnnIn4d, axis=[2])

        # basic cells which is used to build RNN
        numHidden = self.args.rnndim
        cells = [tf.contrib.rnn.LSTMCell(num_units=numHidden, state_is_tuple=True) for _ in range(2)]  # 2 layers

        # stack basic cells
        stacked = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)

        # bidirectional RNN
        # BxTxF -> BxTx2H
        ((fw, bw), _) = tf.nn.bidirectional_dynamic_rnn(cell_fw=stacked, cell_bw=stacked, inputs=rnnIn3d,
                                                        dtype=rnnIn3d.dtype, scope="graph_recognition/bidirectional_rnn")

        # BxTxH + BxTxH -> BxTx2H -> BxTx1X2H
        concat = tf.expand_dims(tf.concat([fw, bw], 2), 2)

        # project output to chars (including blank): BxTx1x2H -> BxTx1xC -> BxTxC
        kernel = tf.Variable(tf.truncated_normal([1, 1, numHidden * 2, len(self.charList) + 1], stddev=0.1))
        logits = tf.squeeze(tf.nn.atrous_conv2d(value=concat, filters=kernel, rate=1, padding='SAME'), axis=[2])
        # with tf.variable_scope('logits'):
        #   logits = tf.squeeze(tf.layers.conv2d(concat, len(self.charList)+1, 1, use_bias=True), axis=[2]) # FIXED BY RONNY
        return logits

    def setupCTC(self, ctcIn3d):
        "create CTC loss and decoder and return them"
        # BxTxC -> TxBxC
        ctcIn3dTBC = tf.transpose(ctcIn3d, [1, 0, 2])
        # ground truth text as sparse tensor
        self.gtTexts = tf.SparseTensor(tf.placeholder(tf.int64, shape=[None, 2]),
                                       tf.placeholder(tf.int32, [None]),
                                       tf.placeholder(tf.int64, [2]))
        # calc loss for batch
        self.seqLen = tf.placeholder(tf.int32, [None])

        loss = tf.nn.ctc_loss(labels=self.gtTexts, inputs=ctcIn3dTBC, sequence_length=self.seqLen,
                              ctc_merge_repeated=True)  # , ignore_longer_outputs_than_inputs=True) #qyk

        # decoder: either best path decoding or beam search decoding
        if self.decoderType == DecoderType.BestPath:
            decoder = tf.nn.ctc_greedy_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen)
        elif self.decoderType == DecoderType.BeamSearch:
            decoder = tf.nn.ctc_beam_search_decoder(inputs=ctcIn3dTBC, sequence_length=self.seqLen, beam_width=50,
                                                    merge_repeated=False)
        elif self.decoderType == DecoderType.WordBeamSearch:
            # import compiled word beam search operation (see https://github.com/githubharald/CTCWordBeamSearch)
            word_beam_search_module = tf.load_op_library('TFWordBeamSearch.so')

            # prepare information about language (dictionary, characters in dataset, characters forming words)
            chars = str().join(self.charList)
            wordChars = open('wordCharList.txt').read().splitlines()[0]
            corpus = open(self.FilePaths.fnCorpus).read()

            # decode using the "Words" mode of word beam search
            decoder = word_beam_search_module.word_beam_search(tf.nn.softmax(ctcIn3dTBC, dim=2), 50, 'Words', 0.0,
                                                               corpus.encode('utf8'), chars.encode('utf8'),
                                                               wordChars.encode('utf8'))

        # return a CTC operation to compute the loss and a CTC operation to decode the RNN output
        return (tf.reduce_mean(loss), decoder)

    def setupWdec(self, args):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():  # all weights count toward weight decay except batchnorm and biases
            if var.op.name.find(r'BatchNorm') == -1 & var.op.name.find(r'bias:0') == -1:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def toSparse(self, texts):
        "put ground truth texts into sparse tensor for ctc_loss"
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for (batchElement, text) in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            labelStr = [self.charList.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(labelStr) > shape[1]:
                shape[1] = len(labelStr)
            # put each label into sparse tensor
            for (i, label) in enumerate(labelStr):
                indices.append([batchElement, i])
                values.append(label)

        return (indices, values, shape)

    def decoderOutputToText(self, ctcOutput):
        "extract texts from output of CTC decoder"
        bt_size = ctcOutput[1].shape[0]  # yike !!!!!!
        # contains string of labels for each batch element
        encodedLabelStrs = [[] for i in range(bt_size)]  # yike self.batchsize !!!!!!!

        # word beam search: label strings terminated by blank
        if self.decoderType == DecoderType.WordBeamSearch:
            blank = len(self.charList)
            for b in range(bt_size):  # yike self.batchsize !!!!!!!
                for label in ctcOutput[b]:
                    if label == blank:
                        break
                    encodedLabelStrs[b].append(label)

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctcOutput[0][0]

            # go over all indices and save mapping: batch -> values
            idxDict = {b: [] for b in range(bt_size)}  # yike self.batchsize !!!!!
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batchElement = idx2d[0]  # index according to [b,t]
                encodedLabelStrs[batchElement].append(label)

        # map labels to chars for all batch elements
        return [str().join([self.charList[c] for c in labelStr]) for labelStr in encodedLabelStrs]

    #########################################################
    ####             Initialize TF (Both)                ####
    #########################################################

    def initTF(self):
        "initialize TF"
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.Session(
            config=tf.ConfigProto(allow_soft_placement=True, gpu_options=tf.GPUOptions(allow_growth=True)))

        #####################################
        ##        SegNet Initiation        ##
        #####################################

        saver_seg = tf.train.Saver(var_list=self.obj_list_savable_seg, max_to_keep=1)  # saver saves model to file

        sess.run(tf.global_variables_initializer())
        print('Ran global_variables_initializer first')
        # Restore from saved model in current checkpoint directory
        latestSnapshot_seg = tf.train.latest_checkpoint(self.args.ckptpath_seg)  # is there a saved model?
        if self.mustRestore_seg and not latestSnapshot_seg:  # if model must be restored (for inference), there must be a snapshot
            raise Exception('No saved model found in: ' + self.args.ckptpath_seg)

        if latestSnapshot_seg:  # load saved model if available
            saver_seg.restore(sess, latestSnapshot_seg)
            print('Init with stored values from ' + latestSnapshot_seg)
        else:
            # sess.run(tf.global_variables_initializer())
            # print('Ran global_variables_initializer')
            sess.run(tf.initializers.variables(var_list=self.var_list_train_seg, name='init_seg'))
            print('Ran initializers.variables on segnet trainable variables')

        '''
            # initialize params from other model (transfer learning)
        if self.args.transfer:
            utils.maybe_download(source_url=self.args.urlTransferFrom,
                                 filename=join(self.args.ckptpath_seg, 'transferFrom'),
                                 target_directory=None,
                                 filetype='folder',
                                 force=True)
            saverTransfer = tf.train.Saver(
                tf.trainable_variables()[:-1])  # load all variables except from logit (classification) layer
            saverTransfer.restore(sess, glob(join(self.args.ckptpath_seg, 'transferFrom', 'model*'))[0].split('.')[0])
            print('Loaded variable values (except logit layer) from ' + self.args.urlTransferFrom)
        '''
        #############################################
        ###         Recognition Initialization    ###
        #############################################
        saver_recg = tf.train.Saver(var_list=self.obj_list_savable_recg, max_to_keep=1)

        latestSnapshot_recg = tf.train.latest_checkpoint(self.args.ckptpath_recg)  # is there a saved model?
        if self.mustRestore_recg and not latestSnapshot_recg:  # if model must be restored (for inference), there must be a snapshot
            raise Exception('No saved model found in: ' + self.args.ckptpath_recg)

        if latestSnapshot_recg:  # load saved model if available
            saver_recg.restore(sess, latestSnapshot_recg)
            print('Init with stored values from ' + latestSnapshot_recg)
        else:
            sess.run(tf.initializers.variables(var_list=self.var_list_train_recg, name='init_recg'))
            print('Ran initializers.variables on recognition trainable variables')
        # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # initialize params from other model (transfer learning)
        """
        if self.args.transfer:
            utils.maybe_download(source_url=self.args.urlTransferFrom,
                                 filename=join(self.args.ckptpath_recg, 'transferFrom'),
                                 target_directory=None,
                                 filetype='folder',
                                 force=True)
            saverTransfer = tf.train.Saver(
                tf.trainable_variables()[:-1])  # load all variables except from logit (classification) layer
            saverTransfer.restore(sess, glob(join(self.args.ckptpath_recg, 'transferFrom', 'model*'))[0].split('.')[0])
            print('Loaded variable values (except logit layer) from ' + self.args.urlTransferFrom)
        """

        return (sess, saver_seg, saver_recg)
    #######################################################
    #####         Training, Inference and Save        #####
    #######################################################

    #######################SegNet##########################
    def saveSeg(self, epoch):
       "save model to file"
       self.saver_seg.save(self.sess, join(self.args.ckptpath_seg, 'model'), global_step=epoch)
    def trainBatchSeg(self, images, labels, labels_recg): # added labels_recg!!!!!!!!!!!!!!!!
        "feed a batch into the NN to train it"

        # sparse = self.toSparse(labels)
        # lrnrate = self.lrInit if self.batchesTrained < self.args.lrDrop1 else (
        # self.lrInit*1e-1 if self.batchesTrained < self.args.lrDrop2 else self.lrInit*1e-2)  # decay learning rate
        bt_size=len(images)
        train_step = self.global_step.eval(session=self.sess)
        sparse = self.toSparse(labels_recg) # added !!!!!!!!!!!!!!!!!!!!!!!!!!!!
        """ fix lr """  ## To Ronny, change the schedule?
        # lr = self.lrInit
        lr = self.lrInit if train_step < self.args.lrDrop1 else (
            self.lrInit * 1e-1 if train_step < self.args.lrDrop2 else self.lrInit * 1e-2)  # yike
        (_, lossValTotal,lossValSeg) = self.sess.run([self.train_op_seg, self.loss_total,self.loss_seg],
                                     {self.input_images_seg: images,
                                      #self.input_images_recg: images, # added !!!!!!!!!!!!!!!
                                      self.input_labels_seg: labels,
                                      self.gtTexts:sparse, # added !!!!!!!!!!!!!!!!!!!!!!!!!!!
                                      self.seqLen: [Model.maxTextLen]*bt_size, #* self.batchsize_seg, #added!!!!!!!!!!!!!!!!!!!!!!
                                      self.learning_rate: lr,
                                      self.phase_train: True,
                                      self.is_training: False})
        # self.batchesTrained += 1
        return lossValTotal,lossValSeg

    def inferBatchSeg(self, imgs):  # modify to compatible to torch. previous def inferBatch(self, batch)
        "feed a batch into the NN to recngnize the texts"

        bt_size = len(imgs)  # yike !!!!!!!!

        pred = self.sess.run(self.pred_seg, feed_dict=
        {self.input_images_seg: imgs,  # check in, comment out in formal run
         # self.input_labels: labels,
         self.phase_train: False,
         self.is_training: False})  # yike self.batchsize!!!!!!!!!
        return pred
    def imageCleanSeg(self, imgs):
        bt_size = len(imgs)
        cleaneds=self.sess.run(self.output_clean_seg, feed_dict={self.input_images_seg: imgs, self.phase_train: False, self.is_training:False})
        return cleaneds.astype('uint8')
    def trainSeg(self, loader, validateloader=None, testloader=None):
        "train NN"
        epoch = 0  # number of training epochs since start
        best_accuracy = 0.0
        step = 0
        while True:
            epoch += 1;
            print('Epoch:', epoch, ' Training...')
            # train
            counter = 0
            # step = 0
            for idx, (images, labels, labels_recg) in enumerate(loader):
                images = images.numpy()
                labels = labels.numpy()
                #labels_recg=labels_recg.numpy()
                loss_value_total,loss_value_seg = self.trainBatchSeg(images, labels, labels_recg)
                assert not np.isnan(loss_value_total), 'Model diverged with loss = NaN'
                step += 1

                if idx % 100 == 0:
                    print('TRAIN: Batch:', idx / len(loader), 'Loss_Total:', loss_value_total)
                    print('TRAIN: Batch:', idx / len(loader), 'Loss_Seg:', loss_value_seg)
                    self.experiment.log_metric('train/loss_total', loss_value_total, step)
                    self.experiment.log_metric('train/loss_seg', loss_value_seg, step)#!!!!!!!!!!!!!!!!!!!!!!!
                    
                    logits = self.sess.run(self.logit_seg,
                                           feed_dict={self.input_images_seg: images,  # check in, comment out in formal run
                                                      # self.input_labels: labels,
                                                      # self.learning_rate: lr,
                                                      self.phase_train: False,
                                                      self.is_training:False})
                    train_acc, train_acc_classes = per_class_acc(logits, labels)  # check in, comment out in formal run

            # train log:
            if self.experiment is not None:
                self.experiment.log_metric('train/acc', train_acc, step)
                self.experiment.log_metric('train/cap_0', train_acc_classes[0], step)
                self.experiment.log_metric('train/cap_1', train_acc_classes[1], step)

            # validate:
            if validateloader != None:
                avg_batch_loss_seg,avg_batch_loss_total, acc_total, cap_0, cap_1,charErrorRate, wordAccuracy = self.validateSeg(validateloader, epoch)
            else:
                avg_batch_loss_seg,avg_batch_loss_total, acc_total, cap_0, cap_1,charErrorRate, wordAccuracy = self.validateSeg(loader, epoch)
            if self.experiment is not None:
                self.experiment.log_metric('valid/acc', acc_total, step)
                self.experiment.log_metric('valid/cap_0', cap_0, step)
                self.experiment.log_metric('valid/cap_1', cap_1, step)
                self.experiment.log_metric('valid/loss_seg', avg_batch_loss_seg, step)
                self.experiment.log_metric('valid/loss_total', avg_batch_loss_total, step)
                
                self.experiment.log_metric('valid/cer', charErrorRate, step)
                self.experiment.log_metric('valid/wer', 1 - wordAccuracy, step)
                
            # test:
            if testloader != None:
                acc_total, cap_0, cap_1 = self.validateSeg(testloader, epoch, is_testing=True)
                if self.experiment is not None:
                    self.experiment.log_metric('test/acc', acc_total, step)
                    self.experiment.log_metric('test/cap_0', cap_0, step)
                    self.experiment.log_metric('test/cap_1', cap_1, step)
                    self.experiment.log_metric('valid/loss_seg', avg_batch_loss_seg, step)
                    self.experiment.log_metric('valid/loss_total', avg_batch_loss_total, step)
                
                    self.experiment.log_metric('valid/cer', charErrorRate, step)
                    self.experiment.log_metric('valid/wer', 1 - wordAccuracy, step)

            # log best metrics
            if acc_total > best_accuracy:  # if best validation accuracy so far, save model parameters
                print('Character error rate improved, save model')
                best_accuracy = acc_total
                noImprovementSince = 0
                self.saveSeg(epoch)
                open(join(self.args.ckptpath_seg, 'accuracy.txt'), 'w').write(
                    'Validation accuracy, class 0, class 1 capture rates of saved model: %f%%, %f%% and %f%% ' % (
                    acc_total * 100.0, cap_0 * 100.0, cap_1 * 100.0))
                if self.experiment is not None:
                    self.experiment.log_metric('best/acc', acc_total, step)
                    self.experiment.log_metric('best/cap_0', cap_0, step)
                    self.experiment.log_metric('best/cap_1', cap_1, step)
            else:
                print('Character error rate not improved')
                noImprovementSince += 1

            # stop training
            if epoch >= self.args.max_epoch: print('Done with training at epoch', epoch,'sigoptObservation=' + str(best_accuracy)); break

    def validateSeg(self, loader, epoch, is_testing=False):
        "validate NN"
        if not is_testing:
            print('Validating NN')
        else:
            print('Testing NN')
        total_val_loss_seg = 0.0
        total_val_loss_total= 0.0
        # num_batches=len(loader)
        hist = np.zeros((self.num_classes, self.num_classes))
        numCharErr, numCharTotal, numWordOK, numWordTotal = 0, 0, 0, 0
        image_upload_count = 0
        for idx, (images, labels, labels_recg) in enumerate(loader):
            
            images = images.numpy()
            labels = labels.numpy()
            bt_size=len(images)            
            sparse=self.toSparse(labels_recg) #added!!!!!!!!!!!!!!!!!!!!!!!!
            #labels_recg=labels_recg.numpy() #added!!!!!!!!!!!!!!!!!!!!!!!!
            #sparse=self.toSparse()
            val_loss_total, val_loss_seg, val_logit,val_images_clean,val_decoded = self.sess.run([self.loss_total,self.loss_seg, self.logit_seg,self.output_clean_seg,self.decoder], feed_dict=
            {self.input_images_seg: images,  # check in, comment out in formal run !!!!!!!!!!!!+++++++val_decoder
             self.input_labels_seg: labels,
             self.gtTexts:sparse,
             self.seqLen: [Model.maxTextLen] * bt_size,#self.batchsize_recg, #changed!!!!!!!!!!!!!!!!!!!
             self.phase_train: False,
             self.is_training:False})  # self.loss,val_loss,

            total_val_loss_seg += val_loss_seg
            total_val_loss_total += val_loss_total#!!!!!!!!!!!!!!!!!!!!!!!
            
            hist += get_hist(val_logit, labels)
            # val_loss=total_val_loss / len(validateloader)*batch_size
            
            #############new text impact record#################
            recognized=self.decoderOutputToText(val_decoded)
            for i in range(len(recognized)):
                numWordOK += 1 if labels_recg[i] == recognized[i] else 0 #batch.gtTexts[i]
                numWordTotal += 1
                dist = editdistance.eval(recognized[i], labels_recg[i])# batch.gtTexts[i])
                numCharErr += dist
                numCharTotal += len(labels_recg[i]) #batch.gtTexts[i]
                ######################################################
                #print(images[i].shape)
                #print(val_images_clean[i].shape)
                if epoch == self.args.max_epoch and image_upload_count < 1000 and self.experiment is not None:
                    text = ' '.join(['[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + labels_recg[i] + '"', '->', '"' + recognized[i] + '"'])
                    im_orig=images[i].astype(int)
                    im_clean=val_images_clean[i].astype(int)
                    im_save=np.transpose(np.concatenate((np.squeeze(im_orig,axis=2),im_clean),axis=1))
                    utils_recg.log_image(self.experiment, im_save, text, 'test-'+('ok' if dist==0 else 'err'), self.args.ckptpath_seg, image_upload_count, epoch)
                    image_upload_count+=1
                    
            '''
            if epoch == self.args.max_epoch and image_upload_count < 1000 and self.experiment is not None:  # decide how many images to upload
                pred = val_logit.argmax(3)
                images = np.squeeze(images, axis=3)
                image_upload_count = log_images_seg_recg(images, pred,labels_recg,recognized, image_upload_count, self.experiment,
                                                self.args.ckptpath_seg)
            '''
        avg_batch_loss_seg = total_val_loss_seg / idx
        avg_batch_loss_total = total_val_loss_total / idx
        
        cls_sample_nums = hist.sum(1).astype(float)
        capture_array = np.diag(hist)
        acc_total = capture_array.sum() / hist.sum()
        capture_rate_ls = []
        for cls in range(self.num_classes):
            if cls_sample_nums[cls] == 0:
                capture_rate = 0.0
            else:
                capture_rate = capture_array[cls] / cls_sample_nums[cls]
            capture_rate_ls.append(capture_rate)
        # iu = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
        # mean_iu=np.nanmean(iu)
        print('VALID: Total accuracy: %f%%. Class 0 capture: %f%%. Class 1 capture: %f%%' % (
            acc_total * 100.0, capture_rate_ls[0] * 100.0, capture_rate_ls[1] * 100.0))
        
        #############new text impact record#################
        charErrorRate = numCharErr / numCharTotal
        wordAccuracy = numWordOK / numWordTotal
        print('VALID: Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))

        ####################################################
        return avg_batch_loss_seg,avg_batch_loss_total, acc_total, capture_rate_ls[0], capture_rate_ls[1],charErrorRate,wordAccuracy

            ##################Recognition##########################

    def trainBatchRecg(self, images, labels): #!!!!!!!!!!!!!!!!

        "feed a batch into the NN to train it"
        sparse = self.toSparse(labels)
        bt_size=len(images)#!!!!!!!!!!!!!!!!!!!!
        lrnrate = self.lrInit if self.batchesTrained < self.args.lrDrop1 else (self.lrInit * 1e-1 if self.batchesTrained < self.args.lrDrop2 else self.lrInit * 1e-2)  # decay learning rate
        (_, lossVal) = self.sess.run([self.optimizer, self.loss_recg],
                                    {self.input_images_recg: images,
                                     self.gtTexts: sparse,
                                     self.seqLen: [Model.maxTextLen]*bt_size,# * self.batchsize_recg,
                                     self.learning_rate: lrnrate,
                                     self.is_training: True})
        self.batchesTrained += 1
        return lossVal

    def inferBatchRecg(self, imgs):  # modify to compatible to torch. previous def inferBatch(self, batch)
        "feed a batch into the NN to recngnize the texts"
        '''if batch to infer less than args.batchsize, error'''

        bt_size = len(imgs)  # yike !!!!!!!!

        decoded = self.sess.run(self.decoder,
                                {self.input_images_recg: imgs, self.seqLen: [Model.maxTextLen] * bt_size,
                                 self.is_training: False})  # yike self.batchsize!!!!!!!!!
        return self.decoderOutputToText(decoded)  # previous batch.imgs
    
    def inferBatchJoint(self,imgs):
        bt_size = len(imgs)  # yike !!!!!!!!

        decoded = self.sess.run(self.decoder,
                                {self.input_images_seg: imgs,
                                 self.seqLen: [Model.maxTextLen] * bt_size,
                                 self.phase_train: False,
                                 self.is_training: False})  # yike self.batchsize!!!!!!!!!

        return self.decoderOutputToText(decoded)  # previous batch.imgs     
    
    def inferCleanBatchJoint(self,imgs):
        bt_size = len(imgs)  # yike !!!!!!!!

        cleaneds,decoded = self.sess.run([self.output_clean_seg,self.decoder],feed_dict=
                                {self.input_images_seg: imgs,
                                 self.seqLen: [Model.maxTextLen] * bt_size,
                                 self.phase_train: False,
                                 self.is_training: False})  # yike self.batchsize!!!!!!!!!

        return cleaneds.astype('uint8'),self.decoderOutputToText(decoded)  # previous batch.imgs     

    def saveRecg(self, epoch):
        "save model to file"
        self.saver_recg.save(self.sess, join(self.args.ckptpath_recg, 'model'), global_step=epoch)

    def trainRecg(self, loader, validateloader=None, testloader=None):  # model

        "train NN"
        epoch = 0  # number of training epochs since start
        bestCharErrorRate = bestWordErrorRate = float('inf')  # best valdiation character error rate

        while True:
            epoch += 1;
            print('Epoch:', epoch, ' Training...')

            # train
            counter = 0
            step = 0

            for idx, (images, labels) in enumerate(loader):

                # convert torchtensor to numpy
                images = images.numpy()

                # train batch
                # try:
                loss = self.trainBatchRecg(images, labels)
                # except:
                #  print(labels)
                step += 1

                # save training status
                if np.mod(idx, 110) == 0:
                    print('TRAIN: Batch:', idx / len(loader), 'Loss:', loss)
                    if self.experiment is not None:
                        self.experiment.log_metric('train/loss', loss, step)

                # log images
                if epoch == 1 and counter < 5:
                    text = labels[counter]
                    utils_recg.log_image(self.experiment, images[counter], text, 'train', self.args.ckptpath_recg, counter, epoch)
                    counter += 1
                # for debug
                # if idx >2:
                #  break

            # validate
            if validateloader != None:
                charErrorRate, wordAccuracy = self.validateRecg(validateloader, epoch)  # yike !!!!!!!!!!!!
            else:  # yike !!!!!!!!!!!!!
                charErrorRate, wordAccuracy = self.validateRecg(loader, epoch)
            if self.experiment is not None:
                self.experiment.log_metric('valid/cer', charErrorRate, step)
                self.experiment.log_metric('valid/wer', 1 - wordAccuracy, step)

            # test
            if testloader != None:
                charErrorRate, wordAccuracy = self.validateRecg(testloader, epoch, is_testing=True)
                if self.experiment is not None:
                    self.experiment.log_metric('test/cer', charErrorRate, step)
                    self.experiment.log_metric('test/wer', 1 - wordAccuracy, step)

            # log best metrics
            if charErrorRate < bestCharErrorRate:  # if best validation accuracy so far, save model parameters
                print('Character error rate improved, save model')
                bestCharErrorRate = charErrorRate
                noImprovementSince = 0
                self.saveRecg(epoch)
                open(join(args.ckptpath_recg, 'accuracy.txt'), 'w').write(
                    'Validation character error rate of saved model: %f%%' % (charErrorRate * 100.0))
            else:
                print('Character error rate not improved')
                noImprovementSince += 1
            if 1 - wordAccuracy < bestWordErrorRate:
                bestWordErrorRate = 1 - wordAccuracy
            if self.experiment is not None:
                self.experiment.log_metric('best/cer', bestCharErrorRate, step)
                self.experiment.log_metric('best/wer', bestWordErrorRate, step)

            # stop training
            if epoch >= args.epochEnd: print('Done with training at epoch', epoch,
                                         'sigoptObservation=' + str(bestCharErrorRate)); break

    def validateRecg(self, loader, epoch, is_testing=False):
        "validate NN"
        if not is_testing: print('Validating NN')
        else: print('Testing NN')
        #loader.validationSet() # comment out by yike. see row 141
        numCharErr, numCharTotal, numWordOK, numWordTotal = 0, 0, 0, 0
        plt.figure(figsize=(6,2))
        counter = 0
        '''
        yike: convert to troch dataloader, test
        '''
        for idx, (images, labels) in enumerate(loader):
            if np.mod(idx,10)==0:
                print(str(idx*50*8))
            images=images.numpy()
            recognized=self.inferBatchRecg(images)

            for i in range(len(recognized)):
                numWordOK += 1 if labels[i] == recognized[i] else 0 #batch.gtTexts[i]
                numWordTotal += 1
                dist = editdistance.eval(recognized[i], labels[i])# batch.gtTexts[i])
                numCharErr += dist
                numCharTotal += len(labels[i]) #batch.gtTexts[i]

                if is_testing and epoch==self.args.epochEnd and self.experiment is not None: #batch.gtTexts[i]
                    text = ' '.join(['[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + labels[i] + '"', '->', '"' + recognized[i] + '"'])
                    utils_recg.log_image(self.experiment, images[i], text, 'test-'+('ok' if dist==0 else 'err'), self.args.ckptpath_recg, counter, epoch)
                    counter += 1 # previous batch.imgs[i]

            if epoch==1 and counter<5 and not is_testing and self.experiment is not None: # log images
                text = ' '.join(['[OK]' if dist == 0 else '[ERR:%d]' % dist, '"' + labels[i] + '"', '->', '"' + recognized[i] + '"'])
                utils_recg.log_image(self.experiment, images[i], text, 'valid', args.ckptpath_recg, counter, epoch) #batch.gtTexts[i]
                counter += 1 #batch.imgs[i]


        # print validation result
        charErrorRate = numCharErr / numCharTotal
        wordAccuracy = numWordOK / numWordTotal
        print('VALID: Character error rate: %f%%. Word accuracy: %f%%.' % (charErrorRate * 100.0, wordAccuracy * 100.0))
        return charErrorRate, wordAccuracy
