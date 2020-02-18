#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : data_ner_tool.py
# @Author: Stoneye
# @Date  : 2019/7/2
# @Desc  :

import tensorflow as tf
import os



class DataProcessing(object):

    def __init__(self,args):

        self.cate1_num=25
        self.cate2_num=3862
        self.max_frame=300
        self.audio_size=128
        self.rgb_size=1024

        self.ground_to_cate1_dict,self.cate1_to_ground_dict = self.get_ground_to_cate1(args.youtube_ground_id_to_cate1_cate2)

    def resize_axis(self,tensor, axis, new_size, fill_value=0):
      """Truncates or pads a tensor to new_size on on a given axis.

      Truncate or extend tensor such that tensor.shape[axis] == new_size. If the
      size increases, the padding will be performed at the end, using fill_value.

      Args:
        tensor: The tensor to be resized.
        axis: An integer representing the dimension to be sliced.
        new_size: An integer or 0d tensor representing the top10w value for
          tensor.shape[axis].
        fill_value: Value to use to fill any top10w entries in the tensor. Will be
          cast to the type of tensor.

      Returns:
        The resized tensor.
      """
      tensor = tf.convert_to_tensor(tensor)
      shape = tf.unstack(tf.shape(tensor))

      pad_shape = shape[:]
      pad_shape[axis] = tf.maximum(0, new_size - shape[axis])

      shape[axis] = tf.minimum(shape[axis], new_size)
      shape = tf.stack(shape)

      resized = tf.concat([
          tf.slice(tensor, tf.zeros_like(shape), shape),
          tf.fill(tf.stack(pad_shape), tf.cast(fill_value, tensor.dtype))
      ], axis)

      # Update shape.
      new_shape = tensor.get_shape().as_list()  # A copy is being made.
      new_shape[axis] = new_size
      resized.set_shape(new_shape)
      return resized
    def parser(self,record):
        feature_keys  = {'id': tf.FixedLenFeature([],tf.string),
                         'labels': tf.VarLenFeature(tf.int64)}
        sequence_features_keys = {'audio': tf.FixedLenSequenceFeature([],tf.string),
                                  'rgb': tf.FixedLenSequenceFeature([],tf.string)}
        context_parsed, sequence_parsed = tf.parse_single_sequence_example(record,feature_keys,sequence_features_keys)
        raw_audio=tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['audio'], tf.uint8), tf.float32),[-1,self.audio_size])
        raw_rgb = tf.reshape(tf.cast(tf.decode_raw(sequence_parsed['rgb'], tf.uint8), tf.float32), [-1, self.rgb_size])

        context_parsed['id'] = tf.expand_dims(context_parsed['id'], -1)
        context_parsed["labels"] = tf.sparse_to_dense(context_parsed["labels"].values, [self.cate2_num], 1,validate_indices=False)

        num_audio_ture_frame=tf.minimum(tf.shape(raw_audio)[0], self.max_frame)
        num_rgb_ture_frame = tf.minimum(tf.shape(raw_rgb)[0], self.max_frame)

        num_audio_rgb_true_frame=tf.maximum(num_audio_ture_frame,num_rgb_ture_frame)
        context_parsed['num_audio_rgb_true_frame']=num_audio_rgb_true_frame

        sequence_parsed['audio'] = self.resize_axis(raw_audio, 0, self.max_frame)
        sequence_parsed['rgb'] = self.resize_axis(raw_rgb, 0, self.max_frame)

        return context_parsed, sequence_parsed

    def input_frame_data(self,frame_path,batch_size=1,num_epoch=1):
        filenames = [os.path.join(frame_path,file) for file in os.listdir(frame_path) if file.endswith('.tfrecord')]
        print("filenames: ",len(filenames))
        dataset  = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(self.parser)
        dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.repeat(num_epoch)
        iterator = dataset.make_one_shot_iterator()
        next_element = iterator.get_next()
        return next_element

    def get_ground_to_cate1(self,in_file):


        print("file: ",in_file)

        ground_to_cate1,cate1_to_ground={},{}
        with open(in_file,'r',encoding='utf-8')as fr:
            for line in fr.readlines():
                line_split=line.rstrip('\n').split('\t')
                if len(line_split)!=3:
                    continue
                raw_id=int(line_split[0])
                cate1_id=int(line_split[1])

                if raw_id not in ground_to_cate1:
                    cate1_to_ground[cate1_id]=raw_id
                    ground_to_cate1[raw_id]=cate1_id



        print("ground_to_cate1: ",len(ground_to_cate1))
        print("cate1_to_ground: ",len(cate1_to_ground))
        return ground_to_cate1,cate1_to_ground

    def get_cate1_cate2_label(self,origin_label):

        batch_cate1_multilabel, batch_cate2_multilabel,batch_origin_cate1, batch_origin_cate2, = [], [], [], []

        for one_sample in origin_label:

            one_cate1_multilabel,one_cate2_multilabel=[],[]
            cate1_list, cate2_list = [], []
            one_origin_label = list(one_sample)
            for token_id in one_origin_label:

                cate1_id = self.ground_to_cate1_dict.get(token_id, 0)
                if cate1_id not in cate1_list:
                    cate1_list.append(cate1_id)



            one_cate1_multilabel = [0.0] * self.cate1_num
            one_cate2_multilabel = [0.0] * self.cate2_num



            for cate1_index in cate1_list:
                one_cate1_multilabel[cate1_index] = 1.0


            for cate2_index in one_origin_label:
                one_cate2_multilabel[cate2_index]=1.0

            # print("one_sample: ",one_sample)
            # print("cate1_list: ",cate1_list)
            # assert 1==2



            batch_cate1_multilabel.append(one_cate1_multilabel)

            batch_cate2_multilabel.append(one_cate2_multilabel)

            batch_origin_cate1.append(cate1_list)
            batch_origin_cate2.append(one_origin_label)


        return batch_cate1_multilabel, batch_cate2_multilabel,batch_origin_cate1, batch_origin_cate2





