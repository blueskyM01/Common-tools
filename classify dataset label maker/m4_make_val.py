import numpy as np
import argparse
import os

'''
介绍：由于glob每次读文件的顺序都是随机的， 因此成的训练集的标签和验证集的标签不一样， 这个模块就是将 ‘验证集的标签’ 与 ‘训练集的标签’ 变为一致
'''
parser = argparse.ArgumentParser()

parser.add_argument("--train_label_dir", default='/media/yang/F/DataSet/ImageNet', type=str, help="data set dir")
parser.add_argument("--train_label_name", default='image_train_jieya.txt', type=str, help="data set name")
parser.add_argument("--val_label_dir", default='/media/yang/F/DataSet/ImageNet', type=str, help="data set label save dir")
parser.add_argument("--val_label_name", default='val.txt', type=str, help="data set label name")
parser.add_argument("--save_file_dir", default='/media/yang/F/DataSet/ImageNet', type=str, help="data set label name")
parser.add_argument("--save_file_name", default='val_t.txt', type=str, help="data set label name")
cfg = parser.parse_args()

def make_val_label(train_label, val_label, savefilename):

    with open(train_label, 'r') as train_label_data:
        counter = 0
        list_1 = []
        for line in train_label_data:
            if line.isspace() == True:
                print('跳过空白行')
            else:
                folder_name, image_name_and_lanel = line.strip().split('/')
                image_name, label = image_name_and_lanel.split('    ')
                if int(label) == counter:
                    counter += 1
                    list_1.append([folder_name, label])
                    # print(list_1)

    f = open(savefilename, 'w+')
    with open(val_label, 'r') as val_label_data:
        for line in val_label_data:
            if line.isspace() == True:
                print('跳过空白行')
            else:
                folder_name, image_name_and_lanel = line.strip().split('/')
                image_name, label = image_name_and_lanel.split('    ')

                for train_foder in list_1:
                    if folder_name in train_foder:
                        label = train_foder[1]
                        f.writelines('\n' + folder_name + '/' + image_name + '    ' + label)


    f.close()


val_label = make_val_label(os.path.join(cfg.train_label_dir, cfg.train_label_name), os.path.join(cfg.val_label_dir, cfg.val_label_name),
                           os.path.join(cfg.save_file_dir, cfg.save_file_name))