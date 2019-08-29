# 介绍
## 1.数据集标注
### 1.1 数据集格式
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
    All_Label_dir = '/media/yang/F/DataSet/detection/pp_dataset/dataset_610_v1_label' # 格式同README.md中的label的存储格式
    All_Image_dir = '/media/yang/F/DataSet/detection/pp_dataset/dataset_610_v1' # 注意： 将所有标注的图像拷到同一文件夹下
    ````