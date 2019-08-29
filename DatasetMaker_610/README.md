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