import os
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict
import time
import cv2

class tfrecords_maker:
    def __init__(self, dataset_dir, dataset_name, label_dir, label_name):
        '''
        :param dataset_dir: 需要生成tfrecords数据集文件所在目录的路径
        :param dataset_name: 要生成tfrecords数据集文件的名称
        :param label_dir: 要生成tfrecords数据集文件label的路径
        :param label_name: 要生成tfrecords数据集文件label的名称
        '''
        self.dataset_name = dataset_name
        self.dataset_dir = dataset_dir
        self.label_dir = label_dir
        self.label_name = label_name

    def convert_to_tfrecord(self, tfrecord_path, tfrecord_folder_name, num_tfrecords, name_tfrecords):
        '''
        :param tfrecord_path: 生成tfrecord文件夹目录
        :param tfrecord_folder_name: 生成tfrecord文件夹名称
        :param num_tfrecords: 生成多少个tfrecord
        :param name_tfrecords: 生成tfrecord文件的名称
        :return:
        '''

        names = np.loadtxt(os.path.join(self.label_dir, self.label_name), dtype=np.str)
        num_image = names.shape[0]
        images_num = int(num_image / num_tfrecords)
        data = self.m4_get_file_label_name(self.label_dir, self.label_name)

        for index_records in range(num_tfrecords):
            if not os.path.exists(os.path.join(tfrecord_path, tfrecord_folder_name)):
                os.makedirs(os.path.join(tfrecord_path, tfrecord_folder_name))
            output_file = os.path.join(tfrecord_path, tfrecord_folder_name,
                                       str(index_records) + '_' + name_tfrecords + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):

                    image_data, label = self.m4_image_lable(data[index], self.dataset_dir, self.dataset_name)

                    if not os.path.exists(image_data):
                        continue
                    with tf.gfile.FastGFile(image_data, 'rb') as file:
                        image = file.read()
                        example = tf.train.Example(features = tf.train.Features(
                            feature = {
                                'image/encoded' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                                'image/label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [label])),
                            }
                        ))
                        record_writer.write(example.SerializeToString())
                        if index % 10 == 0:
                            print('Processed {} of {} images'.format(index + 1, num_image))

        output_file = os.path.join(tfrecord_path, tfrecord_folder_name,
                                   str(num_tfrecords) + '_' + name_tfrecords + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
            for index in range(num_tfrecords * images_num, num_image):
                image_data, label = self.m4_image_lable(data[index], self.dataset_dir, self.dataset_name)
                if not os.path.exists(image_data):
                    continue
                with tf.gfile.FastGFile(image_data, 'rb') as file:
                    image = file.read()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={
                            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                            'image/label': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[label])),
                        }
                    ))
                    record_writer.write(example.SerializeToString())
                    if index % 10 == 0:
                        print('Processed {} of {} images'.format(index + 1, num_image))

        print(output_file + ' is ok....')

    def m4_get_file_label_name(self, label_dir, label_name):
        '''
        :param label_dir: label dir
        :param label_name: label name
        :param dataset_dir: dataset dir
        :param dataset_name: dataset name
        :return:filename_list, label_list
        '''
        filepath_name = os.path.join(label_dir, label_name)
        # save_data_path_name = os.path.join(dataset_dir, dataset_name)
        data = np.loadtxt(filepath_name, dtype=str)
        np.random.shuffle(data)
        return data

    def m4_image_lable(self, iL, dataset_dir, dataset_name):
        save_data_path_name = os.path.join(dataset_dir, dataset_name)
        image_path = os.path.join(save_data_path_name, iL[0].tolist().lstrip("b'").rstrip("'"))
        label = int(iL[1].tolist().lstrip("b'").rstrip("'"))
        return image_path, label





if __name__ == '__main__':
    dataset_dir = '/media/yang/F/DataSet/ImageNet'
    dataset_name = 'image_train_jieya'
    label_dir = '/media/yang/F/DataSet/ImageNet'
    label_name = 'image_train_jieya.txt'
    tfrecord_path = '/media/yang/F/DataSet/ImageNet'
    tfrecord_folder_name = 'tfrecords_imagenet_new'
    name_tfrecords = 'imagenet'
    tensor_file_maker = tfrecords_maker(dataset_dir, dataset_name, label_dir, label_name)
    tensor_file_maker.convert_to_tfrecord(tfrecord_path, tfrecord_folder_name, 1000, name_tfrecords)