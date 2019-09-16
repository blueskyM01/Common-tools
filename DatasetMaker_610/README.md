# 介绍
将所有的“.json”文件汇总成一个总的“.json”文件
## 1.数据集标注
### 1.1 数据集格式
    注意： 将所有的图片合并到同一个文件夹中
    数据集名称
        |--------文件夹1
                    |-----------images
        |--------文件夹2
                    |-----------images
            .
            .
            .
            .
        |--------文件夹n
                    |-----------images
        
### 1.2 label的格式  
    label主文件夹名称
           |----------Face
                        |---------文件夹1
                                    |------json文件（每张图片对应一个json）
                        |---------文件夹2
                                    |------json文件（每张图片对应一个json）
                              .
                              .
                              .      
           |----------UAV
                       |---------文件夹1
                                    |------json文件（每张图片对应一个json）
                       |---------文件夹2
                                    |------json文件（每张图片对应一个json）
                              .
                              .
                              .  
           |----------plan
                        |---------文件夹1
                                    |------json文件（每张图片对应一个json）
                        |---------文件夹2
                                    |------json文件（每张图片对应一个json）
                              .
                              .
                              .  
           |----------parachute
                         |---------文件夹1
                                    |------json文件（每张图片对应一个json）
                         |---------文件夹2
                                    |------json文件（每张图片对应一个json）
                              .
                              .
                              .  
## 2. 最终包含所有label的json文件生成
* 找到[Generate_Final_Json.py](https://github.com/blueskyM01/Common-tools/blob/master/DatasetMaker_610/Generate_Final_Json.py)  
修改对应参数即可：
    ````
    Label_dir = '/media/yang/F/DataSet/detection/mydataset/label' # 格式同README.md中的label的存储格式
    label_file_save_path = '/media/yang/F/DataSet/detection/mydataset/DIY_Label.json'
    image_folder_path = '/media/yang/F/DataSet/detection/mydataset/dataset_610_v1' # 注意： 将所有标注的图像拷到同一文件夹下
    class_list = ['Face', 'UAV', 'Plan', 'Parachute']
    ````
    
## 3. 生成训练label
* 找到[Get_Train_Label.py](https://github.com/blueskyM01/Common-tools/blob/master/DatasetMaker_610/Generate_Final_Json.py)  
修改对应参数即可：
    ````
    json_file = '/media/yang/F/DataSet/detection/mydataset/DIY_Label.json' # 上面生成的总的“json”文件
    save_path = '/media/yang/F/DataSet/detection/mydataset/DIY_Trian_labe.txt' # 生成训练标签的存储路径
    dataset_dir = '/media/yang/F/DataSet/detection/mydataset' # 图像数据集的存储目录
    dataset_name = 'dataset_610_v1' # 图像数据集的名称
    ````