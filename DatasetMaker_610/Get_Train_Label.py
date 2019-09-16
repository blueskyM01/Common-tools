import numpy as np
import os, time, json
from glob import glob
import cv2

def GetTrainLabel(json_file, save_path, dataset_dir, dataset_name):
    image_dir = os.path.join(dataset_dir, dataset_name)
    with open(json_file, 'r') as load_f:
        load_dict = json.load(load_f)

    image_idx = 0
    f = open(save_path, 'w')

    for image_name in load_dict.keys():

        elem = []

        elem.append(str(image_idx))
        image_idx += 1
        boxes = load_dict[image_name]
        image_path = os.path.join(image_dir, image_name)
        elem.append(image_path)
        img = cv2.imread(image_path)
        image_width = img.shape[1]
        image_hegit = img.shape[0]
        elem.append(str(image_width))
        elem.append(str(image_hegit))

        for box in boxes:
            label = box[4]
            x_min = box[0]
            y_min = box[1]
            x_max = box[2]
            y_max = box[3]
            elem.append(str(label))
            elem.append(str(x_min))
            elem.append(str(y_min))
            elem.append(str(x_max))
            elem.append(str(y_max))

        for index in range(len(elem)):
            if index == (len(elem) - 1):
                f.write(elem[index] + '\n')

            else:
                f.write(elem[index] + ' ')
        print('num:', image_idx)
    f.close()


json_file = '/media/yang/F/DataSet/detection/mydataset/DIY_Label.json'
save_path = '/media/yang/F/DataSet/detection/mydataset/DIY_Trian_labe.txt'
dataset_dir = '/media/yang/F/DataSet/detection/mydataset'
dataset_name = 'dataset_610_v1'
GetTrainLabel(json_file, save_path, dataset_dir, dataset_name)