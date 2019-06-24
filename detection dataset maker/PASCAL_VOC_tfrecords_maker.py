import os
import json
import tensorflow as tf
import numpy as np
from collections import defaultdict
import time
import cv2
from glob import glob
import xml.etree.cElementTree as ET
import math
import sys


NAME_LABEL_MAP = {
        'back_ground': 0,
        'aeroplane': 1,
        'bicycle': 2,
        'bird': 3,
        'boat': 4,
        'bottle': 5,
        'bus': 6,
        'car': 7,
        'cat': 8,
        'chair': 9,
        'cow': 10,
        'diningtable': 11,
        'dog': 12,
        'horse': 13,
        'motorbike': 14,
        'person': 15,
        'pottedplant': 16,
        'sheep': 17,
        'sofa': 18,
        'train': 19,
        'tvmonitor': 20
    }


class PASCAL_VOC_Tfrecords_maker:
    def __init__(self, dataset_dir, dataset_name, xml_folder_name, image_folder_name, tfrecords_folder_save_dir,
                 tfrecords_folder_save_name, tfrecords_file_save_name, img_format):
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.xml_folder_name = xml_folder_name
        self.image_folder_name = image_folder_name
        self.tfrecords_folder_save_dir = tfrecords_folder_save_dir
        self.tfrecords_folder_save_name = tfrecords_folder_save_name
        self.tfrecords_file_save_name = tfrecords_file_save_name
        self.img_format = img_format


    def read_xml_gtbox_and_label(self, xml_path):
        """
        :param xml_path: the path of voc xml
        :return: a list contains gtboxes and labels, shape is [num_of_gtboxes, 5],
               and has [lt_x, lt_y, br_x, br_y, y4, label] in a per row
        """

        tree = ET.parse(xml_path)
        root = tree.getroot()
        img_width = None
        img_height = None
        box_list = []
        for child_of_root in root:
            # if child_of_root.tag == 'filename':
            #     assert child_of_root.text == xml_path.split('/')[-1].split('.')[0] \
            #                                  + FLAGS.img_format, 'xml_name and img_name cannot match'

            if child_of_root.tag == 'size':
                for child_item in child_of_root:
                    if child_item.tag == 'width':
                        img_width = int(child_item.text)
                    if child_item.tag == 'height':
                        img_height = int(child_item.text)

            if child_of_root.tag == 'object':
                label = None
                for child_item in child_of_root:
                    if child_item.tag == 'name':
                        label = NAME_LABEL_MAP[child_item.text]
                    if child_item.tag == 'bndbox':
                        tmp_box = []
                        for node in child_item:
                            tmp_box.append(int(node.text))
                        assert label is not None, 'label is none, error'
                        tmp_box.append(label)
                        box_list.append(tmp_box)

        gtbox_label = np.array(box_list, dtype=np.int32)

        return img_height, img_width, gtbox_label

    def convert_pascal_to_tfrecord(self):
        xml_path = os.path.join(self.dataset_dir, self.dataset_name, self.xml_folder_name)
        xml_name_list = glob(xml_path + '/*.xml')

        if not os.path.exists(os.path.join(self.tfrecords_folder_save_dir, self.tfrecords_folder_save_name)):
            os.makedirs(os.path.join(self.tfrecords_folder_save_dir, self.tfrecords_folder_save_name))


        output_file = os.path.join(self.tfrecords_folder_save_dir, self.tfrecords_folder_save_name,
                                   self.tfrecords_file_save_name + '.tfrecord')
        counter = 0
        with tf.python_io.TFRecordWriter(output_file) as record_writer:
            for xml_name in xml_name_list:
                xml = xml_name.replace('\\', '/')

                img_name = xml.split('/')[-1].split('.')[0] + self.img_format
                img_path = os.path.join(self.dataset_dir, self.dataset_name, self.image_folder_name, img_name)
                if not os.path.exists(img_path):
                    print('{} is not exist!'.format(img_path))
                    continue

                img_height, img_width, gtbox_label=self.read_xml_gtbox_and_label(xml_name)


                with tf.gfile.FastGFile(img_path, 'rb') as file:
                    image = file.read()
                    example = tf.train.Example(features=tf.train.Features(
                        feature={'img_name': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_name.encode()])),
                            'img_height': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_height])),
                            'img_width': tf.train.Feature(int64_list=tf.train.Int64List(value=[img_width])),
                            'img': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                            'gtboxes_and_label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[gtbox_label.tostring()])),
                            'num_objects': tf.train.Feature(int64_list=tf.train.Int64List(value=[gtbox_label.shape[0]]))
                        }
                    ))
                    record_writer.write(example.SerializeToString())

                self.view_bar('Conversion progress', counter + 1, len(xml_name_list))
                counter += 1

    def view_bar(self, message, num, total):
        rate = num / total
        rate_num = int(rate * 40)
        rate_nums = math.ceil(rate * 100)
        r = '\r%s:[%s%s]%d%%\t%d/%d' % (message, ">" * rate_num, " " * (40 - rate_num), rate_nums, num, total,)
        sys.stdout.write(r)
        sys.stdout.flush()



if __name__ == '__main__':
    dataset_dir = '/media/yang/F/DataSet/Detection/VOCtrainval_06-Nov-2007/VOCdevkit'
    dataset_name = 'VOC2007'
    xml_folder_name = 'Annotations'
    image_folder_name = 'JPEGImages'
    tfrecords_folder_save_dir = '/media/yang/F/DataSet/Detection/Terecords_VOC'
    tfrecords_folder_save_name = 'VOC_2007'
    tfrecords_file_save_name = 'pascal_train'
    img_format = '.jpg'
    maker = PASCAL_VOC_Tfrecords_maker(dataset_dir, dataset_name, xml_folder_name, image_folder_name,
                                       tfrecords_folder_save_dir, tfrecords_folder_save_name,
                                       tfrecords_file_save_name, img_format)
    maker.convert_pascal_to_tfrecord()