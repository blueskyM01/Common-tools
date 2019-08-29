import os, json, cv2
import numpy as np
from collections import defaultdict
from glob import glob

def Generate_Final_Json(Label_dir):
    final_label = defaultdict(list)  # 创建一个字典，值的type是list
    ClassFolder_list = glob(Label_dir + '/*')
    for class_folder in ClassFolder_list:
        OneLabelFolderList = glob(class_folder + '/*')
        for OneLabelFolder in OneLabelFolderList:
            json_list = glob(OneLabelFolder + '/*')
            for json_file in json_list:
                with open(json_file, 'r') as load_f:
                    load_dict = json.load(load_f)
                    img_name = sorted(load_dict.keys())[0]
                    for box in load_dict[img_name]:
                        final_label[img_name].append(box)
    return final_label


All_Label_dir = '/media/yang/F/DataSet/detection/pp_dataset/dataset_610_v1_label' # 格式同README.md中的label的存储格式
All_Image_dir = '/media/yang/F/DataSet/detection/pp_dataset/dataset_610_v1' # 注意： 将所有标注的图像拷到同一文件夹下

final_json = Generate_Final_Json(All_Label_dir)
classes_list = ['Face', 'UAV', 'Plan', 'Parachute']
for image_name in final_json.keys():
    image_path = os.path.join(All_Image_dir, image_name)
    image = cv2.imread(image_path)
    boxes = final_json[image_name]
    for box in boxes:
        label = box[4]
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])
        cv2.rectangle(image, (x0,y0), (x1,y1), (0,0,255), 2)
        cv2.putText(image, classes_list[label], (x0,y0), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 2)
        cv2.imshow('image_name', image)
        cv2.waitKey(1000)


