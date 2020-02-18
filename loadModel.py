#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : weSeeLoadModel.py
# @Author: Stoneye
# @Date  : 2019/4/26
# @Desc  :


import os
import tensorflow as tf
import numpy as np
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import utils as saved_model_utils


def str2float_and_padding(str, feature_size, max_frame=100, ):
    '''
    rgb|audio 特征 从字符串转化为float
    :param str:
    :return:
    '''

    float_list_x = []

    for i in str[2:-2].replace('[', '').strip().split('],'):
        float_list_x.append(list(map(float, i.strip().split(','))))

    if len(float_list_x) > max_frame:
        float_list_x = float_list_x[:max_frame]
    else:
        for i in range(max_frame - len(float_list_x)):
            float_list_x.append([0.0] * feature_size)
    return float_list_x


def data_iter(in_file):
    for file in os.listdir(in_file):
        if file.endswith('.feature'):
            print("file name: ", file)
            with open(os.path.join(in_file, file), 'r', encoding='utf-8') as f:
                samples = []
                for line in f:
                    one_sample = []
                    line_split = line.strip().split('\t')
                    if len(line_split) != 8:
                        print("%s error line: " % file, len(line_split), line.strip())

                    # vid,cate2_id2,cate1_ids,cate1_media_ids,cate2_media_ids,title,rgb_fea,audio_fea
                    vid = line_split[0]
                    titleId_int_list = list(map(int, line_split[5].split(';')))
                    cate1_label_id = int(line_split[2])
                    cate1_label_oneHot = [0.0] * 19
                    cate1_label_oneHot[cate1_label_id] = 1.0

                    cate2_label_id = int(line_split[1])
                    cate2_label_oneHot = [0.0] * 101
                    cate2_label_oneHot[cate2_label_id] = 1.0

                    cate1_media_id = int(line_split[3])
                    cate2_media_id = int(line_split[4])

                    rgb_fea = line_split[6]
                    audio_fea = line_split[7]

                    rgb_fea_float_list = str2float_and_padding(rgb_fea, 1024, 100)
                    audio_fea_float_list = str2float_and_padding(audio_fea, 128, 100)

                    one_sample.append(vid)
                    one_sample.append(titleId_int_list)
                    one_sample.append(cate1_label_oneHot)
                    one_sample.append(cate2_label_oneHot)
                    one_sample.append(cate1_media_id)
                    one_sample.append(cate2_media_id)
                    one_sample.append(rgb_fea_float_list)
                    one_sample.append(audio_fea_float_list)

                    samples.append(one_sample)
                    if len(samples) % 1 == 0:
                        batch_data = zip(*[samples[index] for index in range(len(samples))])
                        yield batch_data
                        samples = []
                if len(samples) > 0:
                    batch_data = zip(*[samples[index] for index in range(len(samples))])
                    yield batch_data
                    samples = []

def get_input_tensor(sess,pb_path):


    '''
    获得tensor
    :param sess:
    :param pb_path:
    :return:
    '''


    MetaGraphDef = tf.saved_model.loader.load(sess, tags=[tf.saved_model.tag_constants.SERVING],export_dir=pb_path)
    signatureDef_d = MetaGraphDef.signature_def
    signatureDef = signatureDef_d[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    input_firstlabels = signatureDef.inputs['input_firstlabels']
    input_seclabels = signatureDef.inputs['input_seclabels']
    input_media_cate1 = signatureDef.inputs['input_media_cate1']
    input_media_cate2 = signatureDef.inputs['input_media_cate2']
    input_title_ids = signatureDef.inputs['input_title_ids']
    input_video_VidName = signatureDef.inputs['input_video_vidName']
    input_video_RGB_feature = signatureDef.inputs['input_video_RGB_feature']
    input_video_Audio_feature = signatureDef.inputs['input_video_Audio_feature']
    input_dropout_keep_prob = signatureDef.inputs['dropout_keep_prob']

    cate1_top3_predictions = signatureDef.outputs['cate1_top3_predictions']
    cate1_top3_indices = signatureDef.outputs['cate1_top3_indices']
    cate2_top3_predictions = signatureDef.outputs['cate2_top3_predictions']
    cate2_top3_indices = signatureDef.outputs['cate2_top3_indices']


    input_firstlabels_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_firstlabels, sess.graph)
    input_seclabels_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_seclabels, sess.graph)
    input_media_cate1_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_media_cate1, sess.graph)
    input_media_cate2_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_media_cate2, sess.graph)
    input_title_ids_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_title_ids, sess.graph)
    input_video_vidName_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_video_VidName, sess.graph)
    input_video_RGB_feature_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_video_RGB_feature,
                                                                                      sess.graph)
    input_video_Audio_feature_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_video_Audio_feature,
                                                                                        sess.graph)
    input_dropout_keep_prob_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(input_dropout_keep_prob,
                                                                                      sess.graph)

    cate1_top3_predictions_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(cate1_top3_predictions,
                                                                                     sess.graph)
    cate1_top3_indices_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(cate1_top3_indices, sess.graph)
    cate2_top3_predictions_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(cate2_top3_predictions,
                                                                                     sess.graph)
    cate2_top3_indices_tensor = tf.saved_model.utils.get_tensor_from_tensor_info(cate2_top3_indices, sess.graph)

    return input_firstlabels_tensor,input_seclabels_tensor,input_media_cate1_tensor,input_media_cate2_tensor,input_title_ids_tensor, \
           input_video_vidName_tensor,input_video_RGB_feature_tensor,input_video_Audio_feature_tensor,input_dropout_keep_prob_tensor, \
           cate1_top3_predictions_tensor,cate1_top3_indices_tensor,cate2_top3_predictions_tensor,cate2_top3_indices_tensor, \
           cate1_Embedding_256_tensor,cate2_Embedding_256_tensor

def get_data(in_file):

    '''
    valid1.feature:暂时只给了一个case
    :param in_file:
    :return:
    '''

    with open(in_file, 'r', encoding='utf-8') as f:

        for line in f:
            line_split = line.strip().split('\t')
            if len(line_split) != 8:
                print("%s error line: " % in_file, len(line_split), line.strip())

            # vid,cate2_id2,cate1_ids,cate1_media_ids,cate2_media_ids,title,rgb_fea,audio_fea
            vid = line_split[0]
            titleId_int_list = list(map(int, line_split[5].split(';')))
            cate1_label_id = int(line_split[2])
            cate1_label_oneHot = [0.0] * 19
            cate1_label_oneHot[cate1_label_id] = 1.0
            cate2_label_id = int(line_split[1])
            cate2_label_oneHot = [0.0] * 101
            cate2_label_oneHot[cate2_label_id] = 1.0
            cate1_media_id = int(line_split[3])
            cate2_media_id = int(line_split[4])
            rgb_fea = line_split[6]
            audio_fea = line_split[7]
            rgb_fea_float_list = str2float_and_padding(rgb_fea, 1024, 100)
            audio_fea_float_list = str2float_and_padding(audio_fea, 128, 100)


            _vid = np.asarray([vid])
            _titleId_int_list = np.asarray([titleId_int_list])
            _cate1_label_oneHot = np.asarray([cate1_label_oneHot])
            _cate2_label_oneHot = np.asarray([cate2_label_oneHot])
            _cate1_media_id = np.asarray([cate1_media_id])
            _cate2_media_id = np.asarray([cate2_media_id])
            _rgb_fea_float_list = np.asarray([rgb_fea_float_list])
            _audio_fea_float_list = np.asarray([audio_fea_float_list])

        return _vid,_titleId_int_list,_cate1_label_oneHot,_cate2_label_oneHot,_cate1_media_id,_cate2_media_id,_rgb_fea_float_list,_audio_fea_float_list


def infer(_vid,_titleId_int_list,_cate1_label_oneHot,_cate2_label_oneHot,_cate1_media_id,_cate2_media_id,_rgb_fea_float_list,_audio_fea_float_list):


    with tf.Session() as sess:
        input_firstlabels_tensor,input_seclabels_tensor, \
        input_media_cate1_tensor,input_media_cate2_tensor, \
        input_title_ids_tensor,input_video_vidName_tensor, \
        input_video_RGB_feature_tensor, input_video_Audio_feature_tensor, \
        input_dropout_keep_prob_tensor, \
        cate1_top3_predictions_tensor,cate1_top3_indices_tensor, \
        cate2_top3_predictions_tensor,cate2_top3_indices_tensor, \
        cate1_Embedding_256_tensor,cate2_Embedding_256_tensor =get_input_tensor(sess,pb_path='/data/algceph/stoneye/xiaoshipin/cate2/rgb_audio_title/new_mymodel/pb/step_20400')
        #输入参数
        feed = {
            input_firstlabels_tensor: _cate1_label_oneHot,
            input_seclabels_tensor: _cate2_label_oneHot,
            input_media_cate1_tensor: _cate1_media_id,
            input_media_cate2_tensor: _cate2_media_id,
            input_title_ids_tensor: _titleId_int_list,
            input_video_RGB_feature_tensor: _rgb_fea_float_list,
            input_video_vidName_tensor: _vid,
            input_video_Audio_feature_tensor: _audio_fea_float_list,
            input_dropout_keep_prob_tensor: 1.0

        }
        cate1_top3_predictions_val, cate1_top3_indices_val, \
        cate2_top3_predictions_val, cate2_top3_indices_val, \
        cate1_Embedding_256_val, cate2_Embedding_256_val = sess.run(
            [cate1_top3_predictions_tensor, cate1_top3_indices_tensor, cate2_top3_predictions_tensor,
             cate2_top3_indices_tensor, cate1_Embedding_256_tensor, cate2_Embedding_256_tensor], feed_dict=feed)

        return cate1_top3_predictions_val,cate1_top3_indices_val,cate2_top3_predictions_val,cate2_top3_indices_val,cate1_Embedding_256_val,cate2_Embedding_256_val
        #输出

def main():


    _vid, _titleId_int_list, _cate1_label_oneHot, _cate2_label_oneHot, \
    _cate1_media_id, _cate2_media_id, _rgb_fea_float_list, _audio_fea_float_list = get_data('valid1.feature')

    cate1_top3_predictions_val,cate1_top3_indices_val, \
    cate2_top3_predictions_val,cate2_top3_indices_val, \
    cate1_Embedding_256_val, cate2_Embedding_256_val  =infer(_vid,_titleId_int_list,_cate1_label_oneHot,_cate2_label_oneHot,_cate1_media_id,_cate2_media_id,_rgb_fea_float_list,_audio_fea_float_list)

    print('_vid: ', _vid)
    print("cate1_top3_predictions_val: ", cate1_top3_predictions_val)
    print("cate1_top3_indices_val: ", cate1_top3_indices_val)
    print("cate2_top3_predictions_val:", cate2_top3_predictions_val)
    print("cate2_top3_indices_val: ", cate2_top3_indices_val)






if __name__ == '__main__':
    main()

