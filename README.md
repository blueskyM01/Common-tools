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


