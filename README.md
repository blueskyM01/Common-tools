# Introduction
## 1.classify dataset label maker
### 1.1 [m4_Dataset_Label_Maker](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/m4_Dataset_Label_Maker.py)
`功能：`对存储格式为这样的数据集生成'.txt'文件标签  
1) dataset format:
        
        data set folder
                |----------ID folder1
                               |--------xxx.jpg (or)
                               |--------xxx.png (or)
                               |--------xxx.jpeg (or)
                               |--------xxx.bmp
                |----------ID folder2
                               |--------xxx.jpg (or)
                               |--------xxx.png (or)
                               |--------xxx.jpeg (or)
                               |--------xxx.bmp
                               
                             .
                             .
                             .
                        
2) Generated Label .txt file format:

        ID folder/image name    Label
            0000045/015.jpg    0
            0000099/001.jpg    1  
3) `使用方法:` 直接运行 [dataset_label_maker.sh](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/m4_Dataset_Label_Maker.py) 定位到该文件路径下面
    ```
        python m4_Dataset_Label_Maker.py \
        --dataset_dir='/home/yang/study/datasetandparam/parachute' \
        --dataset_name='my_cifar-100' \
        --dataset_label_dir='/home/yang/study/datasetandparam/parachute' \
        --dataset_label_name='my_cifar-100.txt'
    ```
### 1.2 把该数据集转换为tfrecords文件，方便tensorflow训练时读取数据
`功能：`将[m4_Dataset_Label_Maker](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/m4_Dataset_Label_Maker.py) 生成的数据集装换为tfrecords文件
1) 生成文件目录组成
    ```
        tfrecords folder
                    |-------------1_xxx.tfrecords  
                    |-------------2_xxx.tfrecords  
                                .  
                                .
                                .
                                .
    ```
2) `使用方法:` 直接运行 [tfrecords_maker.py](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/tfrecords_maker.py)  
修改对应的参数即可
    ```
        dataset_dir = '/home/yang/study/datasetandparam/parachute'
        dataset_name = 'my_cifar-100'
        label_dir = '/home/yang/study/datasetandparam/parachute'
        label_name = 'my_cifar-100.txt'
        tfrecord_path = '/home/yang/study/datasetandparam/parachute'
        tfrecord_folder_name = 'cifar-100_tfrecords'
        name_tfrecords = 'cifar_image'
        tensor_file_maker = tfrecords_maker(dataset_dir, dataset_name, label_dir, label_name)
        tensor_file_maker.convert_to_tfrecord(tfrecord_path, tfrecord_folder_name, 200, name_tfrecords)                
    ```
### 1.3 [DataSetReader.py](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/DataSetReader.py)
`功能：`tensorflow读取tfrecords文件                                 
1) `使用方法:` 直接运行 [DataSetReader.py](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/DataSetReader.py)  
修改对应的参数即可   
````
    tfrecords_dir = '/home/yang/study/datasetandparam/parachute'
        tfrecords_name = 'cifar-100_tfrecords'
        label_dir = '/home/yang/study/datasetandparam/parachute'
        label_name = 'my_cifar-100.txt'
        class_num = 101
        image_size = (224, 224)
        dataset = Reader(tfrecords_dir, tfrecords_name, label_dir, label_name, class_num, image_size)
        one_element, dataset_size = dataset.build_dataset(batch_size=10, epoch=1, shuffle_num=1000, is_train=True)
    
        with tf.Session() as sess:
            batch_images, batch_labels = sess.run(one_element)
            batch_labels = np.reshape(batch_labels, (10, class_num))
            # print(batch_images, batch_labels)
            count = 0
            for image, label in zip(batch_images, batch_labels):
                count += 1
                image = ((image + 1) * 127.5).astype(np.uint8)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
                print(label)
                cv2.imshow(str(count), image)
            cv2.waitKey(0)
````
### 1.4 做测试时将 `验证集的标签` 与 `训练集的标签` 变为一致
注意： 先要使用[dataset_label_maker.sh](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/m4_Dataset_Label_Maker.py)生成label的`.txt`文件 
1) `使用方法:` 直接运行 [m4_make_val.sh](https://github.com/blueskyM01/Common-tools/blob/master/classify%20dataset%20label%20maker/m4_make_val.sh)  
修改对应的参数即可   
````
    python m4_make_val.py \
    --train_label_dir='/media/yang/F/DataSet/ImageNet' \
    --train_label_name='image_train_jieya.txt' \
    --val_label_dir='/media/yang/F/DataSet/ImageNet' \
    --val_label_name='val.txt' \
    --save_file_dir='/media/yang/F/DataSet/ImageNet' \
    --save_file_name='val_t.txt'
````


## 2. Detection dataset convert to tfrecords 
### 2.1 PASCAL VOC dataset conver to tfrecords
> 注意：img_format的格式，看具体的文件加里面的图像格式(需根据实际情况改动)， 除此之外，这个文件无需在改动  
> 在[PASCAL_VOC_tfrecords_maker.py](https://github.com/blueskyM01/Common-tools/blob/master/detection%20dataset%20maker/PASCAL_VOC_tfrecords_maker.py) 文件中直接运行，修改下面的参数即可

    dataset_dir = '/media/yang/F/DataSet/Detection/VOCtrainval_06-Nov-2007/VOCdevkit'
    dataset_name = 'VOC2007'
    xml_folder_name = 'Annotations'
    image_folder_name = 'JPEGImages'
    tfrecords_folder_save_dir = '/media/yang/F/DataSet/Detection/Terecords_VOC'
    tfrecords_folder_save_name = 'VOC_2007'
    tfrecords_file_save_name = 'pascal_train'
    img_format = '.jpg'
    maker = PASCAL_VOC_Tfrecords_maker(dataset_dir, dataset_name, xml_folder_name, image_folder_name,
                                       tfrecords_folder_save_dir, tfrecords_folder_save_name,
                                       tfrecords_file_save_name, img_format)
    maker.convert_pascal_to_tfrecord()



### 2.2 [DetectDatasetReader.py](https://github.com/blueskyM01/Common-tools/blob/master/detection%20dataset%20maker/DetecteDatasetReader.py) 直接运行该文件
````
注意： 1. batch size只能为 1, 因为每张图片的目标个数不一样，即ground true box的个数不一样。 如果是batch size不为1, tensorflow无法读出
      2. 输出图像的范围为[-1,1], tf.float32型
      3. image_size这个参数未使用
      4. 图像未做预处理
````
修改下面的参数即可  

    tfrecords_dir = '/media/yang/F/DataSet/Detection/Terecords_VOC'
    tfrecords_name = 'VOC_2007'
    image_size = (224, 224)
    dataset = Reader(tfrecords_dir, tfrecords_name, image_size)
    one_element = dataset.build_dataset(batch_size=1, epoch=1, shuffle_num=1000, is_train=False)

    with tf.Session() as sess:
        img_name, image, gtboxes_and_label, num_objects = sess.run(one_element)
        
## 3. COCO数据集的使用  
* 1. 目标检测
    * 训练：`annotations_trainval2017.zip`, `train2017.zip`
        * 解压`annotations_trainval2017.zip`，得到`annotations_trainval2017/annotations`文件夹, 有以下文件：  
            * captions_train2017.json
            * captions_val2017.json
            * instances_train2017.json, 目标检测用， 训练集
            * instances_val2017.json, 目标检测用， 验证集
            * person_keypoints_train2017.json
            * person_keypoints_val2017.json
        * 解压`train2017.zip`，得到`train2017`文件夹， 文件夹下的文件如下：  
            ````
            xxxx.jpg
               .
               .
               .
               .
            xxxx.jpg
            ````
        * instances_train2017.json, instances_val2017.json的文件是如下这个字典：  
            * 可以看出他有5个键： 'info', 'licenses', 'images', 'annotations', 'categories'， 其中'licenses', 'images', 'annotations', 'categories'值的type是list。  
            images数组、annotations数组、categories数组的元素数量是相等的，等于图片的数量。
            ````
                {
                "info": info,
                "licenses": [license], 
                "images": [image],
                "annotations": [annotation],
                "categories": [category]
                }
            ````  
            * 'info'的值也是个字典，如下:
            ````
            info:
                {
                "year": int,
                "version": str,
                "description": str,
                "contributor": str,
                "url": str,
                "date_created": datetime,
                }
            ````
            实例：
            ````
            "info":
                {
                "description":"This is stable 1.0 version of the 2014 MS COCO dataset.",
                "url":"http:\/\/mscoco.org",
                "version":"1.0","year":2014,
                "contributor":"Microsoft COCO group",
                "date_created":"2015-01-27 09:11:52.357475"
                }
            ````
            * 'licenses'的值也是个字典，如下:
            ````
            license:
                {
                "id": int,
                "name": str,
                "url": str,
                }
            ````
            实例：
            ````
            "license":
                {
                "url":"http:\/\/creativecommons.org\/licenses\/by-nc-sa\/2.0\/",
                "id":1,
                "name":"Attribution-NonCommercial-ShareAlike License"
                }
            ````
            * 'images'的值也是个字典，如下:
            ````
            image:
                {
                "id": int,
                "width": int,
                "height": int,
                "file_name": str,
                "license": int,
                "flickr_url": str,
                "coco_url": str,
                "date_captured": datetime,
                }
            ````
            * 'annotations'的值也是个字典，如下:
            ````
            annotation:
                {
                "id": int,
                "image_id": int, # image_id + ‘.jpg’就是图像的名称, 若image_id不足12位数， 则前面补0
                "category_id": int,
                "segmentation": RLE or [polygon],
                "area": float,
                "bbox": [x,y,width,height],
                "iscrowd": 0 or 1,
                }
            ````
            * 'categories'的值也是个字典，如下:
            ````
            category:
                {
                "id": int,
                "name": str,
                "supercategory": str,
                }
            ````
    * 验证
* 2. 制作标签：  修改[COCO_Label_Maker.py]() 中相应的参数，直接运行即可  
    `Generate train.txt/val.txt/test.txt files One line for one image, in the format like：  
     image_index, image_absolute_path, img_width, img_height, box_1, box_2, ... box_n.  
     Box_x format: label_index x_min y_min x_max y_max.
                   (The origin of coordinates is at the left top corner, left top => (xmin, ymin), right bottom => (xmax, ymax).)
     image_index: is the line index which starts from zero.  
     label_index: is in range [0, class_num - 1].  
     For example:  
     0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268  
     1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
    `