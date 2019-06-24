import os
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict
import time
import cv2

class Reader:
    def __init__(self, tfrecords_dir, tfrecords_name, image_size):
        '''
        :param tfrecords_dir: 存储tfrecords文件的文件夹目录
        :param tfrecords_name: 存储tfrecords文件的文件夹名称
        :param label_dir:
        :param label_name:
        :param class_num: 有多少个类别
        :param image_size: 图像的尺寸， 例如[224, 224], 但未使用
        '''


        self.tfrecords_dir = tfrecords_dir    # model_data
        self.tfrecords_name = tfrecords_name
        file_pattern = os.path.join(self.tfrecords_dir, self.tfrecords_name) + "/*" + '.tfrecord'
        self.TfrecordFile = tf.gfile.Glob(file_pattern)
        self.image_size = image_size


    def parser(self, serialized_example):
        """
        Introduction
        ------------
            解析tfRecord数据
        Parameters
        ----------
            serialized_example: 序列化的每条数据
        """
        features = tf.parse_single_example(
            serialized_example,
            features= {'img_name': tf.FixedLenFeature([], dtype = tf.string),
             'img_height': tf.VarLenFeature(dtype = tf.int64),
             'img_width': tf.VarLenFeature(dtype = tf.int64),
             'img': tf.FixedLenFeature([], dtype = tf.string),
             'gtboxes_and_label': tf.FixedLenFeature([], dtype = tf.string),
             'num_objects': tf.VarLenFeature(dtype = tf.int64)
             }
        )
        image = tf.image.decode_png(features['img'], channels = 3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32) * 2.0 - 1.0
        img_height = features['img_height'].values
        img_width = features['img_width'].values
        gtboxes_and_label = tf.decode_raw(features['gtboxes_and_label'], tf.int32)
        gtboxes_and_label = tf.reshape(gtboxes_and_label, [-1, 5])
        num_objects = features['num_objects'].values
        img_name = features['img_name']
        # image = tf.image.resize_images(image, self.image_size)
        return img_name, image, gtboxes_and_label, num_objects




    def build_dataset(self, batch_size, epoch, shuffle_num=10000, is_train=True):
        """
        Introduction
        ------------
            建立数据集dataset
        Parameters
        ----------
            batch_size: batch大小
        Return
        ------
            dataset: 返回tensorflow的dataset
        """

        dataset = tf.data.TFRecordDataset(filenames = self.TfrecordFile)
        dataset = dataset.map(self.parser, num_parallel_calls = 10)
        if is_train:
            dataset = dataset.shuffle(shuffle_num).batch(batch_size).repeat(epoch)
        else:
            dataset = dataset.batch(batch_size).repeat(epoch)
        # dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        one_element = iterator.get_next()
        return one_element

if __name__ == '__main__':
    tfrecords_dir = '/media/yang/F/DataSet/Detection/Terecords_VOC'
    tfrecords_name = 'VOC_2007'
    image_size = (224, 224)
    dataset = Reader(tfrecords_dir, tfrecords_name, image_size)
    one_element = dataset.build_dataset(batch_size=1, epoch=1, shuffle_num=1000, is_train=False)

    with tf.Session() as sess:
        img_name, image, gtboxes_and_label, num_objects = sess.run(one_element)

        # 输出显示部分
        print(img_name)
        print(image.shape)
        print(gtboxes_and_label)
        print(num_objects)

        image = image[0]
        image = (image + 1.0) * 127.5
        image = image.astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for ii in gtboxes_and_label[0]:
            cv2.rectangle(image, (ii[0], ii[1]), (ii[2], ii[3]), (0,0,255), 2)

        cv2.imshow('fff', image)
        cv2.waitKey(0)




        # batch_labels = np.reshape(batch_labels, (10, class_num))
        # # print(batch_images, batch_labels)
        # count = 0
        # for image, label in zip(batch_images, batch_labels):
        #     count += 1
        #     image = ((image + 1) * 127.5).astype(np.uint8)
        #     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
        #     print(label)
        #     cv2.imshow(str(count), image)
        # cv2.waitKey(0)

