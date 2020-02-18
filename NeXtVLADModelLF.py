#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : NeXtVLADModelLF.py
# @Author: stoneye
# @Date  : 2019/05/16
# @Desc  : 
# @license : Copyright(C), Zhenxu Ye
# @Contact : yezhenxu1992@163.com
# @Software : PyCharm

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.slim as slim
import math
import tensorflow.contrib as tc
from tensorflow.python.saved_model import utils as saved_model_utils
from tensorflow.python.saved_model import signature_constants
import shutil
import os

# np.random.seed(5)
# tf.set_random_seed(-1)

BASE=os.path.dirname(os.path.abspath(__file__))


class NeXtVLAD():
    def __init__(self, feature_size, max_frames, cluster_size, is_training=True, expansion=2, groups=None):
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.cluster_size = cluster_size
        self.expansion = expansion    #2
        self.groups = groups          #8


    def forward(self, input, mask=None):
        input = slim.fully_connected(input, self.expansion * self.feature_size, activation_fn=None,
                                     weights_initializer=slim.variance_scaling_initializer())

        attention = slim.fully_connected(input, self.groups, activation_fn=tf.nn.sigmoid,
                                         weights_initializer=slim.variance_scaling_initializer())
        if mask is not None:
            attention = tf.multiply(attention, tf.expand_dims(mask, -1))
        attention = tf.reshape(attention, [-1, self.max_frames*self.groups, 1])
        tf.summary.histogram("sigmoid_attention", attention)
        feature_size = self.expansion * self.feature_size // self.groups

        cluster_weights = tf.get_variable("cluster_weights",
                                          [self.expansion*self.feature_size, self.groups*self.cluster_size],
                                          initializer=slim.variance_scaling_initializer()
                                          )

        # tf.summary.histogram("cluster_weights", cluster_weights)
        reshaped_input = tf.reshape(input, [-1, self.expansion * self.feature_size])
        activation = tf.matmul(reshaped_input, cluster_weights)

        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="cluster_bn",
            fused=False)

        activation = tf.reshape(activation, [-1, self.max_frames * self.groups, self.cluster_size])
        activation = tf.nn.softmax(activation, axis=-1)
        activation = tf.multiply(activation, attention)
        # tf.summary.histogram("cluster_output", activation)
        a_sum = tf.reduce_sum(activation, -2, keepdims=True)

        cluster_weights2 = tf.get_variable("cluster_weights2",
                                           [1, feature_size, self.cluster_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        a = tf.multiply(a_sum, cluster_weights2)

        activation = tf.transpose(activation, perm=[0, 2, 1])

        reshaped_input = tf.reshape(input, [-1, self.max_frames * self.groups, feature_size])
        vlad = tf.matmul(activation, reshaped_input)
        vlad = tf.transpose(vlad, perm=[0, 2, 1])
        vlad = tf.subtract(vlad, a)

        vlad = tf.nn.l2_normalize(vlad, 1)

        vlad = tf.reshape(vlad, [-1, self.cluster_size * feature_size])
        vlad = slim.batch_norm(vlad,
                center=True,
                scale=True,
                is_training=self.is_training,
                scope="vlad_bn",
                fused=False)

        return vlad


class NeXtVLADModelLF(object):

    def __init__(self, args,isTraining):



        self.is_training=isTraining
        self.lr = args.lr
        self.cluster_size = 128
        self.groups=8
        self.expansion=2
        self.NetVLADHiddenSize = 2048

        self.max_frames = 300
        self.audio_feature_size = 128
        self.rgb_feature_size = 1024
        self.num_classes = 3862
        self._init_vocab_and_emb(args)
        self._init_placeholder()
        self._build_graph()

        self.global_step = tf.Variable(0, trainable=False, name='global_step')  # 系统维护的一个全集变量，每过一个step（mini-batch），自动+1
        self.learning_rate = tf.train.exponential_decay(
            learning_rate=self.lr,
            global_step=self.global_step,
            decay_steps=1000000,
            decay_rate=0.8,
            staircase=True)

        # Adds update_ops (e.g., moving average updates in batch normalization) as
        # a dependency to the train_op.

        self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.total_loss,
                                                                                 global_step=self.global_step)  # 采用Adam优化器进行梯度下降的参数更新迭代

    def _init_vocab_and_emb(self, args):
        print("_init_vocab_and_emb is processing")
        self.youtu_8m_cate1_dict = self._get_cate_dict(args.youtu_8m_cate1_dict)
        print("len(self.cate1_dict): ", len(self.youtu_8m_cate1_dict))
        self._random_embedding()

    def _random_embedding(self):
        self.youtu_8m_cate1_embedding = np.random.standard_normal([len(self.youtu_8m_cate1_dict), 128])

    def _init_placeholder(self):

        print("youtu_8m_cate1_dict:",len(self.youtu_8m_cate1_dict))
        self.input_cate1_multilabel = tf.placeholder(tf.float32, shape=[None, len(self.youtu_8m_cate1_dict)])
        self.input_cate2_multilabel = tf.placeholder(tf.float32, shape=[None, self.num_classes])
        self.input_video_vidName = tf.placeholder(tf.string, shape=[None,1])
        self.input_video_RGB_feature = tf.placeholder(tf.float32, shape=[None, self.max_frames, self.rgb_feature_size])
        self.input_video_Audio_feature = tf.placeholder(tf.float32,
                                                        shape=[None, self.max_frames, self.audio_feature_size])
        self.input_rgb_audio_true_frame= tf.placeholder(tf.int32,shape=(None,))
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

    def _get_cate_dict(self, file):
        token2dict = {}
        with open(file, 'r', encoding='utf-8')as fr:
            for line in fr.readlines():
                line_split = line.rstrip('\n').split('\t')
                if len(line_split) != 2:
                    print("error getTagDict: ", len(line_split), line.strip())
                    continue
                token = line_split[0]
                id = line_split[1]
                if token not in token2dict:
                    token2dict[token] = id

        print("token2dict: ",len(token2dict))
        return token2dict

    def calculate_loss(self, predictions, labels):
        with tf.name_scope("loss_xent"):
            epsilon = 1e-8
            float_labels = tf.cast(labels, tf.float32)
            cross_entropy_loss = float_labels * tf.log(predictions + epsilon) + (
                    1 - float_labels) * tf.log(1 - predictions + epsilon)
            cross_entropy_loss = tf.negative(cross_entropy_loss)
            return tf.reduce_mean(tf.reduce_sum(cross_entropy_loss, 1))

    def _build_graph(self):


        hidden1_size = self.NetVLADHiddenSize
        gating_reduction = 8
        model_input = tf.concat([self.input_video_RGB_feature, self.input_video_Audio_feature],-1)  # [batch,max_frame,1024+128]
        mask = tf.sequence_mask(self.input_rgb_audio_true_frame, 300, dtype=tf.float32)
        max_frames = model_input.get_shape().as_list()[1]
        video_nextvlad = NeXtVLAD(1024, max_frames, self.cluster_size, self.is_training, groups=self.groups,
                                  expansion=self.expansion)
        audio_nextvlad = NeXtVLAD(128, max_frames, self.cluster_size // 2, self.is_training, groups=self.groups // 2,
                                  expansion=self.expansion)

        with tf.variable_scope("video_VLAD"):
            vlad_video = video_nextvlad.forward(model_input[:, :, 0:1024], mask=mask)

        with tf.variable_scope("audio_VLAD"):
            vlad_audio = audio_nextvlad.forward(model_input[:, :, 1024:], mask=mask)

        vlad = tf.concat([vlad_video, vlad_audio], 1)

        vlad = slim.dropout(vlad, keep_prob=self.dropout_keep_prob, is_training=self.is_training, scope="vlad_dropout")

        vlad_dim = vlad.get_shape().as_list()[1]
        print("VLAD dimension", vlad_dim)
        hidden1_weights = tf.get_variable("hidden1_weights",
                                          [vlad_dim, hidden1_size],
                                          initializer=slim.variance_scaling_initializer())

        activation = tf.matmul(vlad, hidden1_weights)
        activation = slim.batch_norm(
            activation,
            center=True,
            scale=True,
            is_training=self.is_training,
            scope="hidden1_bn",
            fused=False)


        gating_weights_1 = tf.get_variable("gating_weights_1",
                                           [hidden1_size, hidden1_size // gating_reduction],
                                           initializer=slim.variance_scaling_initializer())

        gates = tf.matmul(activation, gating_weights_1)

        gates = slim.batch_norm(
            gates,
            center=True,
            scale=True,
            is_training=self.is_training,
            activation_fn=slim.nn.relu,
            scope="gating_bn")

        gating_weights_2 = tf.get_variable("gating_weights_2",
                                           [hidden1_size // gating_reduction, hidden1_size],
                                           initializer=slim.variance_scaling_initializer()
                                           )
        gates = tf.matmul(gates, gating_weights_2)

        gates = tf.sigmoid(gates)
        tf.summary.histogram("final_gates", gates)

        activation = tf.multiply(activation, gates)

        l2_penalty = 1e-8

        with tf.variable_scope("output_cate1"):
            self.cate1_logits = slim.fully_connected(
                activation,len(self.youtu_8m_cate1_dict), activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                biases_regularizer=slim.l2_regularizer(l2_penalty),
                weights_initializer=slim.variance_scaling_initializer())
            self.cate1_probs = tf.nn.sigmoid(self.cate1_logits)

            self.cate1_top5_probs_value, self.cate1_top5_probs_index = tf.nn.top_k(self.cate1_probs, 5)

            # self.total_loss=self.calculate_loss(predictions=self.logits,labels=self.input_cate2_multilabel)

            self.cate1_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_cate1_multilabel,
                                                                      logits=self.cate1_logits,
                                                                      name="cate2_cross_loss")



            self.mean_cate1_loss = tf.reduce_mean(self.cate1_loss)



        self.cate1_embeddings = tf.cast(self.youtu_8m_cate1_embedding, dtype=tf.float32)


        with tf.variable_scope('attention'):
            self.U = tf.tanh(tc.layers.fully_connected(self.cate1_embeddings, num_outputs=512,
                                                       activation_fn=None, biases_initializer=None)
                             + tc.layers.fully_connected(tf.expand_dims(activation, 1),
                                                         num_outputs=512,
                                                         activation_fn=None))
            self.first_logits = tc.layers.fully_connected(self.U, num_outputs=1, activation_fn=None)
            self.first_scores = tf.nn.softmax(self.first_logits, 1)  # [batch,]



            self.cate1_embeddings_attention = tf.reduce_sum(self.cate1_embeddings * self.first_scores,axis=1)  # [batch,max_len,2h]



        with tf.variable_scope("output_cate2"):
            self.cate2_logits = slim.fully_connected(
                tf.concat([activation,self.cate1_embeddings_attention],-1), 3862, activation_fn=None,
                weights_regularizer=slim.l2_regularizer(l2_penalty),
                biases_regularizer=slim.l2_regularizer(l2_penalty),
                weights_initializer=slim.variance_scaling_initializer())
            self.cate2_probs = tf.nn.sigmoid(self.cate2_logits)

            self.cate2_top20_probs_value, self.cate2_top20_probs_index = tf.nn.top_k(self.cate2_probs, 20)
            self.cate2_top40_probs_value, self.cate2_top40_probs_index = tf.nn.top_k(self.cate2_probs, 40)

            # self.total_loss=self.calculate_loss(predictions=self.logits,labels=self.input_cate2_multilabel)

            self.cate2_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_cate2_multilabel,
                                                                      logits=self.cate2_logits,
                                                                      name="cate2_cross_loss")
            self.mean_cate2_loss = tf.reduce_mean(self.cate2_loss)

        self.total_loss = self.mean_cate1_loss + 2 * self.mean_cate2_loss



    def _savePb(self, sess, pbPath):

        '''
        pbPath的构造： pb_model_dir = "{}/step_{}".format(args.pb_savePath, model.global_step.eval())
        :param sess:
        :param pbPath:
        :return:
        '''

        print("pb save path: ", pbPath)
        builder = tf.saved_model.builder.SavedModelBuilder(pbPath)

        inputs = {"input_cate1_multilabel": saved_model_utils.build_tensor_info(self.input_cate1_multilabel),
                  "input_cate2_multilabel": saved_model_utils.build_tensor_info(self.input_cate2_multilabel),
                  "input_video_vidName": saved_model_utils.build_tensor_info(self.input_video_vidName),
                  "input_video_RGB_feature": saved_model_utils.build_tensor_info(self.input_video_RGB_feature),
                  "input_video_Audio_feature": saved_model_utils.build_tensor_info(self.input_video_Audio_feature),
                  "dropout_keep_prob": saved_model_utils.build_tensor_info(self.dropout_keep_prob)
                  }

        outputs = {
            "cate1_top3_predictions": saved_model_utils.build_tensor_info(self.cate1_top3_probs_value),
            "cate1_top3_indices": saved_model_utils.build_tensor_info(self.cate1_top3_probs_index),
            "cate2_top20_predictions": saved_model_utils.build_tensor_info(self.cate2_top20_probs_value),
            "cate2_top20_indices": saved_model_utils.build_tensor_info(self.cate2_top20_probs_index),

        }

        signature = tf.saved_model.signature_def_utils.build_signature_def(inputs, outputs,
                                                                           method_name=signature_constants.PREDICT_METHOD_NAME)
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature})

        builder.save()

    def _rmMorePb(self, pbPathDir, max_pb=3):
        '''
        删除多余的pb目录，只保留max_pb_num个
        :param pbPathDir: args.pb_savePath
        :param max_pb_num: 3
        :return:
        '''
        result = []
        for pb_dir_name in os.listdir(pbPathDir):
            if pb_dir_name.startswith('step_'):
                result.append(pb_dir_name)
        result.sort(key=lambda x: int(x.split('_')[1]), reverse=True)
        if len(result) > max_pb:
            delPbList = result[max_pb:]
            for pb_dir in delPbList:
                shutil.rmtree(os.path.join(pbPathDir, pb_dir))
