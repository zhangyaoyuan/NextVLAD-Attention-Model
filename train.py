#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : offline_train.py
# @Author: Stoneye
# @Date  : 2019/7/2
# @Desc  :

from NeXtVLADModelLF import NeXtVLADModelLF
from  data_util import DataProcessing
import numpy as np
import tensorflow as tf
import argparse
import os
import eval_util
import subprocess

from tensorflow.python.saved_model import utils as saved_model_utils

#
# np.random.seed(5)
# tf.set_random_seed(-1)

parser = argparse.ArgumentParser()




parser.add_argument("-b", "--batch_size", help="the size of batch ", default=80)  # mini-batch的数目
parser.add_argument("-lr", "--lr", help="learning rate", default=0.0002)  # 学习率
parser.add_argument("-cate1_vocab", "--youtu_8m_cate1_dict", help="the data path of tag_cate1_vocab",
                    default="/data/ceph_11015/ssd/stoneye/youtube_ucd/model/version_1/vocab/youtube_cate1.txt")  # tag_cate1_vocab

parser.add_argument("-youtube_ground_id_to_cate1_cate2", "--youtube_ground_id_to_cate1_cate2", help="the data path of tag_cate2_vocab",
                    default="/data/ceph_11015/ssd/stoneye/youtube_ucd/model/version_1/vocab/youtube_ground_id_to_cate1_cate2.txt")  # tag_cate2_vocab


parser.add_argument("-train_path", "--train_path", help="the data path of train", default="/data/ceph_11015/ssd/stoneye/youtube_ucd/data/train")  # 训练目录
parser.add_argument("-valid_path", "--valid_path", help="the data path of valid", default="/data/ceph_11015/ssd/stoneye/youtube_ucd/data/part_valid")  # 验证集合，用来挑选保存最好的模型
parser.add_argument("-rf", "--report_freq", help="frequency to report loss", default=500,type=int)  # 打印频率，每隔300个step，即经过300个mini-batch,输出训练的loss
parser.add_argument("-vf", "--valid_freq", help="frequency to do validation", default=1000, type=int)  # 验证频率，每隔300个step，即经过300个mini-batch,验证当前验证集合的loss
parser.add_argument("-hd", "--hidden_size", help="the units of hidden", default=128)  # 隐藏层的数目
parser.add_argument("-e", "--epoch", help="the number of epoch", default=5)  # 轮数，即训练集合过几遍模型。
parser.add_argument("-m", "--model_type", help="the model for [train | predict_test | predict_valid]", default="train")  # 控制开关，即训练or测试
parser.add_argument("-md", "--model_dir", help="the dir of save model", default="/data/ceph_11015/ssd/stoneye/youtube_ucd/model/version_1/attention_cascade_bm/mymodel")  # 保存ckpt模型的路径
parser.add_argument("-pb", "--pb_savePath", help="the dir of save model", default="/data/ceph_11015/ssd/stoneye/youtube_ucd/model/version_1/attention_cascade_bm/mymodel/pb")  # 保存pb模型的路径
parser.add_argument("-test_path", "--test_path", help="the data path of test",default="/data/ceph_11015/ssd/stoneye/youtube_ucd/data/test")  # 测试集路径

parser.add_argument("-predict_out_path", "--predict_out_path", help="the data path of predict_out",default="/data/ceph_11015/ssd/stoneye/youtube_ucd/model/version_1/attention_cascade_bm/predict_out/testSet.txt")  # 测试文件的预测输出路径
parser.add_argument("-predict_cate1", "--predict_cate1", help="the data path of predict_out",default="/data/ceph_11015/ssd/stoneye/youtube_ucd/model/version_1/attention_cascade_bm/predict_out/validSet.cate1.txt")  # 测试文件的预测输出路径
parser.add_argument("-predict_cate2", "--predict_cate2", help="the data path of predict_out",default="/data/ceph_11015/ssd/stoneye/youtube_ucd/model/version_1/attention_cascade_bm/predict_out/validSet.cate2.txt")  # 测试文件的预测输出路径


args = parser.parse_args()


def creat_model(session, args, isTraining):
    '''
        模型计算图的构建
        :param session:            会话实例
        :param args:               配置参数
        :param word_id_vocab:      分词字典     token:id
        :param first_tag2id_dict:  一级标签字典  token:id
        :param second_tag2id_dict: 二级标签字典  token:id
        :param third_tag2id_dict:  三级标签字典  token:id
        :return: _model：模型对象
        '''
    _model = NeXtVLADModelLF(args, isTraining)
    # app_tag_model函数： 模型计算图的具体实现
    ckpt = tf.train.get_checkpoint_state(args.model_dir)

    # 判断是否存在模型
    if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        # print("Reloading model parameters..")
        print("Reading model parameters from {}".format(ckpt.model_checkpoint_path))
        saver = tf.train.Saver(tf.global_variables())
        saver.restore(sess=session, save_path=ckpt.model_checkpoint_path)  # 调用saver接口，将各个tensor变量的值赋给对应的tensor
        print(session.run(_model.global_step))
    else:
        if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)
        print("Created new model parameters..")
        session.run(tf.global_variables_initializer())
    return _model


def train():
    '''
        模型，训练任务
        :return:
    '''
    valid_max_accuracy = -9999
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:

        data_object = DataProcessing(args)
        train_batch_example = data_object.input_frame_data(frame_path=args.train_path, batch_size=args.batch_size,
                                                           num_epoch=args.epoch)

        model = creat_model(sess, args, isTraining=True)  # 构建模型计算图
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=3)  # max_to_keep 表征只保留最好的3个模型
        print("Begin training..")

        train_cate1_perr_list = []
        train_cate1_gap_list = []
        train_cate2_perr_list = []
        train_cate2_gap_list = []
        train_total_loss_list = []

        try:
            while True:
                context_parsed, sequence_parsed = sess.run(train_batch_example)

                batch_origin_labels = [np.nonzero(row)[0].tolist() for row in context_parsed['labels']]
                cate1_multilabel, cate2_multilabel, batch_origin_cate1, batch_origin_cate2 = data_object.get_cate1_cate2_label(
                    batch_origin_labels)
                batch_vid_name = np.asarray(context_parsed['id'])
                batch_num_audio_rgb_true_frame = np.asarray(context_parsed['num_audio_rgb_true_frame'])
                batch_cate1_label_multiHot = np.asarray(cate1_multilabel)  # batch,cate1_nums
                batch_cate2_label_multiHot = np.asarray(cate2_multilabel)  # batch,cate2_nums
                batch_rgb_fea_float_list = np.asarray(sequence_parsed['rgb'])  # batch,max_frame,1024
                batch_audio_fea_float_list = np.asarray(sequence_parsed['audio'])  # batch,max_frame,1
                # print("batch_vid_name.shape:",batch_vid_name.shape)
                # print("batch_cate1_label_multiHot.shape: ",batch_cate1_label_multiHot.shape)
                # print("batch_cate2_label_multiHot.shape: ",batch_cate2_label_multiHot.shape)
                # print("batch_rgb_fea_float_list.shape: ",batch_rgb_fea_float_list.shape)
                # print("batch_audio_fea_float_list.shape: ",batch_audio_fea_float_list.shape)
                # print("batch_num_audio_rgb_true_frame: ",batch_num_audio_rgb_true_frame)
                # print("batch_num_audio_rgb_true_frame: ", np.asarray(batch_num_audio_rgb_true_frame).shape)
                # assert 1==2

                feed = dict(
                    zip([model.input_video_vidName, model.input_cate1_multilabel, model.input_cate2_multilabel,
                         model.input_video_RGB_feature, model.input_video_Audio_feature,
                         model.input_rgb_audio_true_frame, model.dropout_keep_prob],
                        [batch_vid_name, batch_cate1_label_multiHot, batch_cate2_label_multiHot,
                         batch_rgb_fea_float_list, batch_audio_fea_float_list, batch_num_audio_rgb_true_frame, 0.5]))

                cate1_probs, cate2_probs, total_loss, _ = sess.run(
                    [model.cate1_probs, model.cate2_probs, model.total_loss, model.optimizer], feed)

                train_cate1_perr = eval_util.calculate_precision_at_equal_recall_rate(cate1_probs,
                                                                                      batch_cate1_label_multiHot)
                train_cate1_gap = eval_util.calculate_gap(cate1_probs, batch_cate1_label_multiHot)
                train_cate2_perr = eval_util.calculate_precision_at_equal_recall_rate(cate2_probs,
                                                                                      batch_cate2_label_multiHot)
                train_cate2_gap = eval_util.calculate_gap(cate2_probs, batch_cate2_label_multiHot)

                train_cate1_perr_list.append(train_cate1_perr)
                train_cate1_gap_list.append(train_cate1_gap)
                train_cate2_perr_list.append(train_cate2_perr)
                train_cate2_gap_list.append(train_cate2_gap)

                train_total_loss_list.append(total_loss)

                if model.global_step.eval() % args.report_freq == 0:
                    print("report_freq: ", args.report_freq)

                    print(
                        'cate1_train: Step:{} ; aver_train_cate1_perr:{} ; aver_train_cate1_gap_list:{} ; aver_total_loss:{}'.format(
                            model.global_step.eval(),
                            1.0 * np.sum(train_cate1_perr_list) / len(train_cate1_perr_list),
                            1.0 * np.sum(train_cate1_gap_list) / len(train_cate1_gap_list),
                            1.0 * np.sum(train_total_loss_list) / len(train_total_loss_list))
                    )

                    print(
                        'cate2_train: Step:{} ; aver_train_cate2_perr:{} ; aver_train_cate2_gap_list:{} ; aver_total_loss:{}'.format(
                            model.global_step.eval(),
                            1.0 * np.sum(train_cate2_perr_list) / len(train_cate2_perr_list),
                            1.0 * np.sum(train_cate2_gap_list) / len(train_cate2_gap_list),
                            1.0 * np.sum(train_total_loss_list) / len(train_total_loss_list))
                    )

                    train_cate1_perr_list = []
                    train_cate1_gap_list = []
                    train_cate2_perr_list = []
                    train_cate2_gap_list = []
                    train_total_loss_list = []

                if model.global_step.eval() >=10000 and model.global_step.eval() % args.valid_freq == 0:
                    # 统计验证集的准确率
                    print("valid infer is process  111!")
                    print('model.global_step.eval(): ', model.global_step.eval())

                    valid_cate2_gap_aver_loss = eval()
                    # 保存当前验证集合准确率最高的模型
                    if valid_cate2_gap_aver_loss > valid_max_accuracy:
                        print("save the model, step= : ", model.global_step.eval())
                        valid_max_accuracy = valid_cate2_gap_aver_loss
                        checkpoint_path = os.path.join(args.model_dir, 'model.ckpt')
                        saver.save(sess=sess, save_path=checkpoint_path, global_step=model.global_step.eval())


        except tf.errors.OutOfRangeError:
            print("train processing is finished!")


def eval():
    g1 = tf.Graph()
    with g1.as_default():
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:

            data_object = DataProcessing(args)

            valid_batch_example = data_object.input_frame_data(frame_path=args.valid_path, batch_size=160, num_epoch=1)
            model = creat_model(sess, args, isTraining=False)  # 构建模型计算图

            # app_tag_model函数： 模型计算图的具体实现
            ckpt = tf.train.get_checkpoint_state(args.model_dir)

            # 判断是否存在模型
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                # print("Reloading model parameters..")
                print("valid step: Reading model parameters from {}".format(ckpt.model_checkpoint_path))
                saver = tf.train.Saver(tf.global_variables())
                saver.restore(sess=sess, save_path=ckpt.model_checkpoint_path)  # 调用saver接口，将各个tensor变量的值赋给对应的tensor
                print(sess.run(model.global_step))
            else:
                if not os.path.exists(args.model_dir):
                    os.makedirs(args.model_dir)
                print("valid step: Created new model parameters..")
                sess.run(tf.global_variables_initializer())

            valid_cate2_perr_list = []
            valid_cate2_gap_list = []
            valid_total_loss_list = []

            try:
                while True:
                    context_parsed, sequence_parsed = sess.run(valid_batch_example)
                    batch_origin_labels = [np.nonzero(row)[0].tolist() for row in context_parsed['labels']]
                    cate1_multilabel, cate2_multilabel, batch_origin_cate1, batch_origin_cate2 = data_object.get_cate1_cate2_label(
                        batch_origin_labels)
                    batch_vid_name = np.asarray(context_parsed['id'])
                    batch_cate1_label_multiHot = np.asarray(cate1_multilabel)  # batch,cate1_nums
                    batch_cate2_label_multiHot = np.asarray(cate2_multilabel)  # batch,cate2_nums
                    batch_rgb_fea_float_list = np.asarray(sequence_parsed['rgb'])  # batch,max_frame,1024
                    batch_audio_fea_float_list = np.asarray(sequence_parsed['audio'])  # batch,max_frame,128
                    batch_num_audio_rgb_true_frame = np.asarray(context_parsed['num_audio_rgb_true_frame'])

                    feed = dict(
                        zip([model.input_video_vidName, model.input_cate1_multilabel, model.input_cate2_multilabel,
                             model.input_video_RGB_feature, model.input_video_Audio_feature,
                             model.input_rgb_audio_true_frame,
                             model.dropout_keep_prob],
                            [batch_vid_name, batch_cate1_label_multiHot, batch_cate2_label_multiHot,
                             batch_rgb_fea_float_list, batch_audio_fea_float_list, batch_num_audio_rgb_true_frame,
                             1.0]))

                    cate2_probs, total_loss = sess.run([model.cate2_probs, model.total_loss], feed)

                    cate2_perr = eval_util.calculate_precision_at_equal_recall_rate(cate2_probs,
                                                                                    batch_cate2_label_multiHot)
                    cate2_gap = eval_util.calculate_gap(cate2_probs, batch_cate2_label_multiHot)

                    valid_cate2_perr_list.append(cate2_perr)
                    valid_cate2_gap_list.append(cate2_gap)

                    valid_total_loss_list.append(total_loss)

            except tf.errors.OutOfRangeError:
                print("end!")

                valid_cate2_perr_aver_loss = 1.0 * np.sum(valid_cate2_perr_list) / len(valid_cate2_perr_list)
                valid_cate2_gap_aver_loss = 1.0 * np.sum(valid_cate2_gap_list) / len(valid_cate2_gap_list)

                valid_total_valid_aver_loss = 1.0 * np.sum(valid_total_loss_list) / len(valid_total_loss_list)

                print('total valid cate2_perr_aver_loss: %0.4f' % valid_cate2_perr_aver_loss)
                print('total valid cate2_gap_aver_loss: %0.4f' % valid_cate2_gap_aver_loss)
                print('***********************')
                print('total valid total_valid_aver_loss: %0.4f' % valid_total_valid_aver_loss)

            return valid_cate2_gap_aver_loss


def remove_file(file_path):
    command = "rm {}".format(file_path)
    subprocess.call(command, shell=True)


def predict_valid():
    if os.path.exists(args.predict_cate1):
        remove_file(args.predict_cate1)

    if os.path.exists(args.predict_cate2):
        remove_file(args.predict_cate2)

    with open(args.predict_cate1, 'a+') as fout1:
        fout1.write("VideoId,LabelConfidencePairs" + '\n')

    with open(args.predict_cate2, 'a+') as fout2:
        fout2.write("VideoId,LabelConfidencePairs" + '\n')

    data_object = DataProcessing(args)

    test_batch_example = data_object.input_frame_data(frame_path=args.valid_path,
                                                      batch_size=200, num_epoch=1)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = creat_model(sess, args, isTraining=False)  # 构建模型计算图

        num = 0
        try:
            while True:

                batch_cate1, batch_cate2, batch_vid, cate1_batch_index, cate1_batch_value, cate2_batch_index, cate2_batch_value = [], [], [], [], [], [], []

                context_parsed, sequence_parsed = sess.run(test_batch_example)

                batch_origin_labels = [np.nonzero(row)[0].tolist() for row in context_parsed['labels']]
                cate1_multilabel, cate2_multilabel, batch_origin_cate1, batch_origin_cate2 = data_object.get_cate1_cate2_label(
                    batch_origin_labels)
                batch_vid_name = np.asarray(context_parsed['id'])
                batch_cate1_label_multiHot = np.asarray(cate1_multilabel)  # batch,cate1_nums
                batch_cate2_label_multiHot = np.asarray(cate2_multilabel)  # batch,cate2_nums
                batch_rgb_fea_float_list = np.asarray(sequence_parsed['rgb'])  # batch,max_frame,1024
                batch_audio_fea_float_list = np.asarray(sequence_parsed['audio'])  # batch,max_frame,128
                batch_num_audio_rgb_true_frame = np.asarray(context_parsed['num_audio_rgb_true_frame'])

                feed = dict(
                    zip([model.input_video_vidName, model.input_cate1_multilabel, model.input_cate2_multilabel,
                         model.input_video_RGB_feature, model.input_video_Audio_feature,
                         model.input_rgb_audio_true_frame,
                         model.dropout_keep_prob],
                        [batch_vid_name, batch_cate1_label_multiHot, batch_cate2_label_multiHot,
                         batch_rgb_fea_float_list, batch_audio_fea_float_list, batch_num_audio_rgb_true_frame, 1.0]))

                cate1_top5_probs_value, cate1_top5_probs_index, cate2_top40_probs_index, cate2_top40_probs_value = sess.run(
                    [model.cate1_top5_probs_value, model.cate1_top5_probs_index, model.cate2_top40_probs_index,
                     model.cate2_top40_probs_value], feed)

                num += 1
                print("testSet step_num: ", num)

                for one_batch_vid in batch_vid_name:
                    batch_vid.append(list(one_batch_vid))

                for one_batch_index in cate1_top5_probs_index:
                    cate1_batch_index.append(list(one_batch_index))

                for one_batch_value in cate1_top5_probs_value:
                    cate1_batch_value.append(list(one_batch_value))

                for one_batch_cate1 in batch_origin_cate1:
                    batch_cate1.append(one_batch_cate1)

                with open(args.predict_cate1, 'a+') as fout1:

                    for vid, cate1, index, value in zip(
                            *[batch_vid, batch_cate1, cate1_batch_index, cate1_batch_value]):
                        one_result = []
                        tmp_index_vale_list = []
                        vid = vid[0].decode(encoding='utf-8')
                        one_result.append(' '.join(map(str, cate1)))
                        one_result.append('\t')
                        one_result.append(vid)
                        one_result.append(',')
                        for k, v in zip(*[index, value]):
                            tmp_index_vale_list.append(str(k))
                            tmp_index_vale_list.append(str("%.6f" % v))

                        one_result.append(' '.join(tmp_index_vale_list))

                        # print("one_result: ",''.join(one_result))
                        fout1.write(''.join(one_result) + '\n')

                for one_batch_index in cate2_top40_probs_index:
                    cate2_batch_index.append(list(one_batch_index))

                for one_batch_value in cate2_top40_probs_value:
                    cate2_batch_value.append(list(one_batch_value))

                for one_batch_cate2 in batch_origin_cate2:
                    batch_cate2.append(one_batch_cate2)

                with open(args.predict_cate2, 'a+') as fout2:

                    for vid, cate2, index, value in zip(
                            *[batch_vid, batch_cate2, cate2_batch_index, cate2_batch_value]):
                        one_result = []
                        tmp_index_vale_list = []
                        vid = vid[0].decode(encoding='utf-8')
                        one_result.append(' '.join(map(str, cate2)))
                        one_result.append('\t')
                        one_result.append(vid)
                        one_result.append(',')
                        for k, v in zip(*[index, value]):
                            tmp_index_vale_list.append(str(k))
                            tmp_index_vale_list.append(str("%.6f" % v))

                        one_result.append(' '.join(tmp_index_vale_list))

                        # print("one_result: ",''.join(one_result))
                        fout2.write(''.join(one_result) + '\n')

        except tf.errors.OutOfRangeError:
            print("train processing is finished!")


def predict_test():
    if os.path.exists(args.predict_out_path):
        remove_file(args.predict_out_path)

    with open(args.predict_out_path, 'a+') as fout1:
        fout1.write("VideoId,LabelConfidencePairs" + '\n')

    data_object = DataProcessing(args)
    test_batch_example = data_object.input_frame_data(frame_path=args.test_path,
                                                      batch_size=200, num_epoch=1)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)  # 如果有gpu，则优先用gpu，否则走cpu资源
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = creat_model(sess, args, isTraining=False)  # 构建模型计算图

        num = 0
        try:
            while True:
                batch_vid, batch_index, batch_value = [], [], []
                context_parsed, sequence_parsed = sess.run(test_batch_example)
                batch_origin_labels = [np.nonzero(row)[0].tolist() for row in context_parsed['labels']]
                cate1_multilabel, cate2_multilabel, batch_origin_cate1, batch_origin_cate2 = data_object.get_cate1_cate2_label(
                    batch_origin_labels)
                batch_vid_name = np.asarray(context_parsed['id'])
                batch_cate1_label_multiHot = np.asarray(cate1_multilabel)  # batch,cate1_nums
                batch_cate2_label_multiHot = np.asarray(cate2_multilabel)  # batch,cate2_nums
                batch_rgb_fea_float_list = np.asarray(sequence_parsed['rgb'])  # batch,max_frame,1024
                batch_audio_fea_float_list = np.asarray(sequence_parsed['audio'])  # batch,max_frame,128
                batch_num_audio_rgb_true_frame = np.asarray(context_parsed['num_audio_rgb_true_frame'])

                feed = dict(
                    zip([model.input_video_vidName, model.input_cate1_multilabel, model.input_cate2_multilabel,
                         model.input_video_RGB_feature, model.input_video_Audio_feature,
                         model.input_rgb_audio_true_frame,
                         model.dropout_keep_prob],
                        [batch_vid_name, batch_cate1_label_multiHot, batch_cate2_label_multiHot,
                         batch_rgb_fea_float_list, batch_audio_fea_float_list, batch_num_audio_rgb_true_frame, 1.0]))

                cate2_top20_probs_index, cate2_top20_probs_value = sess.run(
                    [model.cate2_top20_probs_index, model.cate2_top20_probs_value], feed)

                num += 1
                print("validSet step_num: ", num)
                for one_batch_index in cate2_top20_probs_index:
                    batch_index.append(list(one_batch_index))

                for one_batch_value in cate2_top20_probs_value:
                    batch_value.append(list(one_batch_value))

                for one_batch_vid in batch_vid_name:
                    batch_vid.append(list(one_batch_vid))

                with open(args.predict_out_path, 'a+') as fout2:

                    for vid, index, value in zip(*[batch_vid, batch_index, batch_value]):
                        one_result = []
                        tmp_index_vale_list = []
                        vid = vid[0].decode(encoding='utf-8')
                        one_result.append(vid)
                        one_result.append(',')
                        for k, v in zip(*[index, value]):
                            tmp_index_vale_list.append(str(k))
                            tmp_index_vale_list.append(str("%.6f" % v))

                        one_result.append(' '.join(tmp_index_vale_list))

                        # print("one_result: ",''.join(one_result))
                        fout2.write(''.join(one_result) + '\n')

                batch_vid, batch_index, batch_value = [], [], []



        except tf.errors.OutOfRangeError:
            print("train processing is finished!")


def main():
    if args.model_type == "train":
        train()
    elif args.model_type == "predict_test":
        predict_test()

    elif args.model_type == "predict_valid":
        predict_valid()


if __name__ == '__main__':
    main()