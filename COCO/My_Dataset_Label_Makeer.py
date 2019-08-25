import numpy as np
import os, time, json
from glob import glob
import cv2

def m4_GetLabel(file_dir, save_path, dataset_dir, dataset_name):
    image_dir = os.path.join(dataset_dir, dataset_name)
    filenale_list = glob(file_dir + '/*')

    image_idx = 5011
    f = open(save_path, 'w')
    for j_name in filenale_list:
        elem = []
        with open(j_name, 'r') as load_f:
            load_dict = json.load(load_f)
        elem.append(str(image_idx))
        image_idx += 1
        image_name = sorted(load_dict.keys())[0]
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









m4_GetLabel('/media/yang/F/DataSet/detection/pp_dataset/P_dataset_Label', '/media/yang/F/DataSet/detection/my_dateset.txt',
            '/media/yang/F/DataSet/detection/pp_dataset',
            'p_dataset')