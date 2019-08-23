import os, json, cv2
import numpy as np
from collections import defaultdict
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
def read_annotation(dataset_dir, dataset_name, label_folder, save_path):
    '''
    功能： Generate train.txt/val.txt/test.txt files One line for one image, in the format like：
          image_index, image_absolute_path, img_width, img_height, box_1, box_2, ... box_n.
          Box_x format: label_index x_min y_min x_max y_max.
                        (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
          image_index: is the line index which starts from zero.
          label_index: is in range [0, class_num - 1].
          For example:
          0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
          1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
    :param dataset_dir: 具体的图像文件夹存储的目录
    :param dataset_name: 图像文件夹的名称
    :param label_folder:
    :param save_path: 生成的“.txt"存储的路径
    :return:
    '''
    f = open(save_path, 'w')
    counter = 0

    xml_path = os.path.join(dataset_dir, dataset_name, label_folder)
    xml_name_list = glob(xml_path + '/*.xml')
    for xml_name in xml_name_list:
        xml = xml_name.replace('\\', '/')
        img_name = xml.split('/')[-1].split('.')[0] + '.jpg'
        img_path = os.path.join(dataset_dir, dataset_name, 'JPEGImages', img_name)
        if not os.path.exists(img_path):
            print('{} is not exist!'.format(img_path))
            continue



        img_height, img_width, gtbox_label = read_xml_gtbox_and_label(xml_name)


        elem = []
        elem.append(counter) # image_index
        counter += 1
        elem.append(img_path) # image_absolute_path


        elem.append(img_width)
        elem.append(img_height)

        boxes = []
        for info in gtbox_label:
            x_min = info[0]
            y_min = info[1]
            x_max = info[2]
            y_max = info[3]
            boxes.append(info[4])
            boxes.append(x_min)
            boxes.append(y_min)
            boxes.append(x_max)
            boxes.append(y_max)

        elem = elem + boxes
        for index in range(len(elem)):
            if index == 1:
                f.write(elem[index] + ' ')

            elif index == (len(elem) - 1):
                f.write(str(round(elem[index], 2)) + '\n')

            else:
                f.write(str(round(elem[index], 2)) + ' ')
        print('num:', counter)
    f.close()


def read_xml_gtbox_and_label(xml_path):
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


read_annotation(dataset_dir='/media/yang/F/DataSet/detection/VOCTest',
                dataset_name='VOC2007',
                label_folder='Annotations',
                save_path='/media/yang/F/DataSet/detection/val.txt')