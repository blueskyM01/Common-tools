import os, json, cv2
import numpy as np
from collections import defaultdict
from glob import glob

class Final_All_label_Maker:
    def __init__(self, Label_dir, label_file_save_path, image_folder_path, class_list):
        '''
        功能： 将所有的“.json”文件汇总成一个总的“.json”文件
        :param Label_dir: 存储label的目录路径： 即 readme中 1.2 label的格式“label主文件夹”的路径
        :param label_file_save_path: 生成总的".json"文件路径
        :param image_folder_path: 存储图像的文件夹路径： 注意：要将所有的图像拷贝的同一个文件夹下
        :param class_list: 类别名称的列表
        '''
        self.Label_dir = Label_dir
        self.label_file_save_path = label_file_save_path
        self.image_folder_path = image_folder_path
        self.class_list = class_list

    def Generate_Final_Json(self):
        final_label = defaultdict(list)  # 创建一个字典，值的type是list
        ClassFolder_list = glob(self.Label_dir + '/*')
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

    def ShowAnnoResult(self):
        final_json = self.Generate_Final_Json()
        for image_name in final_json.keys():
            image_path = os.path.join(self.image_folder_path, image_name)
            image = cv2.imread(image_path)
            boxes = final_json[image_name]
            for box in boxes:
                label = box[4]
                x0 = int(box[0])
                y0 = int(box[1])
                x1 = int(box[2])
                y1 = int(box[3])
                cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 255), 2)
                cv2.putText(image, self.class_list[label], (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow('image_name', image)
            cv2.waitKey(2000)

    def Get_JSON_File(self):
        final_json = self.Generate_Final_Json()
        with open(self.label_file_save_path, "w") as f:
            json.dump(final_json, f)




Label_dir = '/media/yang/F/DataSet/detection/mydataset/label' # 格式同README.md中的label的存储格式
label_file_save_path = '/media/yang/F/DataSet/detection/mydataset/DIY_Label.json'
image_folder_path = '/media/yang/F/DataSet/detection/mydataset/dataset_610_v1' # 注意： 将所有标注的图像拷到同一文件夹下
class_list = ['Face', 'UAV', 'Plan', 'Parachute']

label_maker = Final_All_label_Maker(Label_dir, label_file_save_path, image_folder_path, class_list)
# label_maker.ShowAnnoResult() # 查看标注结果
label_maker.Get_JSON_File()



