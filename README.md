# Introduction
## 1.classify dataset label maker
### 1.1 [m4_Dataset_Label_Maker](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/m4_Dataset_Label_Maker.py)
对存储格式为这样的数据集生成'.txt'文件标签  
1) dataset format:
        
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
3) 使用方法[dataset_label_maker.sh](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/m4_Dataset_Label_Maker.py) 定位到该文件路径下面
    ```
    python m4_Dataset_Label_Maker.py \
    --dataset_dir='/home/yang/study/datasetandparam/parachute' \
    --dataset_name='my_cifar-100' \
    --dataset_label_dir='/home/yang/study/datasetandparam/parachute' \
    --dataset_label_name='my_cifar-100.txt'
    ```
### 1.2 把该数据集转换为tfrecords文件，方便tensorflow训练时读取数据
1) 


