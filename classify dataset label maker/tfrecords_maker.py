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
        image_data, labels = self.m4_get_file_label_name(self.label_dir, self.label_name, self.dataset_dir, self.dataset_name)

        for index_records in range(num_tfrecords):
            if not os.path.exists(os.path.join(tfrecord_path, tfrecord_folder_name)):
                os.makedirs(os.path.join(tfrecord_path, tfrecord_folder_name))
            output_file = os.path.join(tfrecord_path, tfrecord_folder_name,
                                       str(index_records) + '_' + name_tfrecords + '.tfrecords')
            with tf.python_io.TFRecordWriter(output_file) as record_writer:
                for index in range(index_records * images_num, (index_records + 1) * images_num):
                    with tf.gfile.FastGFile(image_data[index], 'rb') as file:
                        image = file.read()
                        example = tf.train.Example(features = tf.train.Features(
                            feature = {
                                'image/encoded' : tf.train.Feature(bytes_list = tf.train.BytesList(value = [image])),
                                'image/label' : tf.train.Feature(int64_list = tf.train.Int64List(value = [labels[index]])),
                            }
                        ))
                        record_writer.write(example.SerializeToString())
                        if index % 10 == 0:
                            print('Processed {} of {} images'.format(index + 1, num_image))
                print(output_file + ' is ok....')

    def m4_get_file_label_name(self, label_dir, label_name, dataset_dir, dataset_name):
        '''
        :param label_dir: label dir
        :param label_name: label name
        :param dataset_dir: dataset dir
        :param dataset_name: dataset name
        :return:filename_list, label_list
        '''
        filepath_name = os.path.join(label_dir, label_name)
        save_data_path_name = os.path.join(dataset_dir, dataset_name)
        data = np.loadtxt(filepath_name, dtype=str)
        filename = data[:, 0].tolist()
        label = data[:, 1].tolist()
        filename_list = []
        label_list = []
        for i in range(data.shape[0]):
            filename_list.append(os.path.join(save_data_path_name, filename[i].lstrip("b'").rstrip("'")))
            label_list.append(int(label[i].lstrip("b'").rstrip("'")))
        return filename_list, label_list




if __name__ == '__main__':
    dataset_dir = '/home/yang/study/datasetandparam/parachute'
    dataset_name = 'my_cifar-100'
    label_dir = '/home/yang/study/datasetandparam/parachute'
    label_name = 'my_cifar-100.txt'
    tfrecord_path = '/home/yang/study/datasetandparam/parachute'
    tfrecord_folder_name = 'cifar-100_tfrecords'
    name_tfrecords = 'cifar_image'
    tensor_file_maker = tfrecords_maker(dataset_dir, dataset_name, label_dir, label_name)
    tensor_file_maker.convert_to_tfrecord(tfrecord_path, tfrecord_folder_name, 200, name_tfrecords)