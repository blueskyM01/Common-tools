import numpy as np
import os
from glob import glob
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--dataset_dir", default='dir', type=str, help="data set dir")
parser.add_argument("--dataset_name", default='name', type=str, help="data set name")
parser.add_argument("--dataset_label_dir", default='label_dir', type=str, help="data set label save dir")
parser.add_argument("--dataset_label_name", default='label_name', type=str, help="data set label name")
cfg = parser.parse_args()

def m4_face_label_maker(filepath,savefilename):
    '''
    1) dataset format:
        dataset dir
            |---------data set name
                            |----------ID folder
                                           |--------xxx.jpg (or)
                                           |--------xxx.png (or)
                                           |--------xxx.jpeg (or)
                                           |--------xxx.bmp
    2) Generated Label .txt file format:
        ID folder/image name    Label
            0000045/015.jpg    0
            0000099/001.jpg    1
    :param filepath: dataset dir + dataset name
    :param savefilename: name of saved label .txt file
    :return:
    '''
    namelist = os.listdir(filepath)
    filename = filepath + '/'
    labelall = []
    idx = 0
    for name in namelist:
        imagename = []
        foldername = filename + name
        imagename = imagename + glob(foldername + '/*.jpg') + glob(foldername + '/*.png') \
               + glob(foldername + '/*.jpeg') + glob(foldername + '/*.bmp')

        for i in range(len(imagename)):
            label = [name,imagename[i].split('/')[-1],idx]
            labelall.append(label)
        print(idx)
        idx +=1
    f = open(savefilename,'w+')
    for j in range(len(labelall)):
        f.writelines('\n'+str(labelall[j][0])+'/'+str(labelall[j][1])+ '    ' + str(labelall[j][2]))
    f.close()

m4_face_label_maker(os.path.join(cfg.dataset_dir,cfg.dataset_name),os.path.join(cfg.dataset_label_dir,cfg.dataset_label_name))