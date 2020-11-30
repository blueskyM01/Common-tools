import tensorflow as tf
import numpy as np
import visualize_anchors as vis
def enum_ratios(anchors, anchor_ratios):
    '''
    功能： 求9个anchor的 w 和 h
    ratio = h /w
    :param anchors: [[128, 128, 128, 128],
                     [256, 256, 256, 256],
                     [512, 512, 512, 512]]
    :param anchor_ratios: [0.5, 1., 2.]
    :return: 每个anchor的 w 和 h
             ws:[[128/sqrt(0.5)], [256/sqrt(0.5)], [512/sqrt(0.5)], [128/sqrt(1.0)], [256/sqrt(1.0)], [512/sqrt(1.0)], [128/sqrt(2.0)], [256/sqrt(2.0)], [512/sqrt(2.0)]]
             shape: [9,1]
             hs:[[128×sqrt(0.5)], [256×sqrt(0.5)], [512×sqrt(0.5)], [128×sqrt(1.0)], [256×sqrt(1.0)], [512×sqrt(1.0)], [128×sqrt(2.0)], [256×sqrt(2.0)], [512×sqrt(2.0)]]
             shape: [9,1]
    '''
    ws = anchors[:, 2]  # for base anchor: w == h
    hs = anchors[:, 3]

    sqrt_ratios = tf.sqrt(tf.constant(anchor_ratios))
    '''
    原数组A的shape： (2,3)
    A[:, tf.newaxis]的shape： (2,1,3)
    A[tf.newaxis, :]的shape： (1,2,3)

    原数组B的shape： (3,)
    B[:, tf.newaxis]的shape： (3,1)
    '''
    '''
    sqrt_ratios[:, tf.newaxis]: [[sqrt(0.5)], 
                                 [sqrt(1.0)],
                                 [sqrt(2.0)]]
    sqrt_ratios[:, tf.newaxis]广播后变为：[[sqrt(0.5), sqrt(0.5), sqrt(0.5)],
                                         [sqrt(1.0), sqrt(1.0), sqrt(1.0)],
                                         [sqrt(2.0), sqrt(2.0), sqrt(2.0)]]
    ws/sqrt_ratios[:, tf.newaxis]: [[128/sqrt(0.5), 256/sqrt(0.5), 512/sqrt(0.5)],
                                    [128/sqrt(1.0), 256/sqrt(1.0), 512/sqrt(1.0)],
                                    [128/sqrt(2.0), 256/sqrt(2.0), 512/sqrt(2.0)]] 
    ws最终reshape变为[[128/sqrt(0.5)], [256/sqrt(0.5)], [512/sqrt(0.5)], [128/sqrt(1.0)], [256/sqrt(1.0)], [512/sqrt(1.0)], [128/sqrt(2.0)], [256/sqrt(2.0)], [512/sqrt(2.0)]]

    hs同理          [[128×sqrt(0.5)], [256×sqrt(0.5)], [512×sqrt(0.5)], [128×sqrt(1.0)], [256×sqrt(1.0)], [512×sqrt(1.0)], [128×sqrt(2.0)], [256×sqrt(2.0)], [512×sqrt(2.0)]]
    '''
    ws = tf.reshape(ws / sqrt_ratios[:, tf.newaxis], [-1, 1])
    hs = tf.reshape(hs * sqrt_ratios[:, tf.newaxis], [-1, 1])

    return hs, ws

def enum_scales(base_anchor, anchor_scales):
    '''
    :param base_anchor: 一个anchor， [0, 0, 256, 256]
    :param anchor_scales: [0.5, 1., 2.]
    :return: anchor_scales： 面积的尺度， 每个元素的平方就是面积。
                          [[128, 128, 128, 128],
                           [256, 256, 256, 256],
                           [512, 512, 512, 512]]
            '''

    '''
    广播机制：
    1. 最后一个维度为1，都可以进行广播；
    2. anchor_scales广播变成：
                            [[0.5, 0.5, 0.5, 0.5],
                             [1.0, 1.0, 1.0, 1.0],
                             [2.0, 2.0, 2.0, 2.0]]
    3. base_anchor与anchor_scales广播后的点成，得：
                                                [[128, 128, 128, 128],
                                                 [256, 256, 256, 256],
                                                 [512, 512, 512, 512]]
    '''
    anchor_scales = base_anchor * tf.constant(anchor_scales, dtype=tf.float32, shape=(len(anchor_scales), 1))

    return anchor_scales

def make_anchors(base_anchor_size, anchor_scales, anchor_ratios,
                 featuremap_height, featuremap_width,
                 stride):
    '''
    功能：为每个中心点生成anchor， feature map 映射回原图为中心， 每个中心生成9个anchor。 共： 9 × featuremap_width × featuremap_height个,
        每个anchor表示为：[xmin, ymin. xmax, ymax]。
        按照从上像下，从左到右的顺序(第一个点生成9个anchor， 然后下一个点生成9个anchor, ....直到最后)
    :param base_anchor_size: 256
    :param anchor_scales: [0.5, 1., 2.0]
    :param anchor_ratios: [0.5, 1., 2.0]
    :param featuremap_height:
    :param featuremap_width:
    :param stride: [16]
    :return: anchors: [-1, 4], feature map 映射回原图为中心， 每个中心生成9个anchor。 共： 9 × featuremap_width × featuremap_height个,
                      每个anchor表示为：[xmin, ymin. xmax, ymax]
    '''

    base_anchor = tf.constant([0, 0, base_anchor_size, base_anchor_size], tf.float32)  # [x_center, y_center, w, h] [0,0,256,256]

    ws, hs = enum_ratios(enum_scales(base_anchor, anchor_scales),
                         anchor_ratios)  # per locations ws and hs
    '''
    假设： featuremap_width=3， tf.range(featuremap_width, dtype=tf.float32)： [0,1,2]
    x_centers=[0,1,2,3] * [16] = [0,16,32]
    假设： featuremap_width=2， tf.range(featuremap_width, dtype=tf.float32)： [0,1]
    y_centers=[0,1] * [16] = [0,16]
    '''
    x_centers = tf.range(featuremap_width, dtype=tf.float32) * stride
    y_centers = tf.range(featuremap_height, dtype=tf.float32) * stride

    '''
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)
    x_centers: [[0,16,32],
                [0,16,32]]
    y_centers: [[ 0, 0,  0],
                [16,16, 16]]
    '''
    x_centers, y_centers = tf.meshgrid(x_centers, y_centers)

    '''
    tf.meshgrid(ws, x_centers)
    1. 如果ws， x_centers的shape不是(n,)这样的格式， 则先拍平变成(n,)这样的格式
    2. 后面的处理跟上面一样
    ws: [[w1, ...,w9],
         [w1, ...,w9],
         [w1, ...,w9],
         [w1, ...,w9],
         [w1, ...,w9],
         [w1, ...,w9]] 6行， 9列
    x_centers: [[ 0,  0, ...,  0],
                [16, 16, ..., 16],
                [32, 32, ..., 32],
                [ 0,  0, ...,  0],
                [16, 16, ..., 16]
                [32, 32, ..., 32]] 6行， 9列
    hs: [[h1, ...,h9],
         [h1, ...,h9],
         [h1, ...,h9],
         [h1, ...,h9],
         [h1, ...,h9],
         [h1, ...,h9]] 6行， 9列
    y_centers: [[ 0,  0, ..., 0],
                [ 0,  0, ..., 0],
                [ 0,  0, ..., 0],
                [16, 16, ..., 0],
                [16, 16, ..., 0],
                [16, 16, ..., 0]]6行，9列           
    '''
    ws, x_centers = tf.meshgrid(ws, x_centers)
    hs, y_centers = tf.meshgrid(hs, y_centers)

    '''
    anchor_centers:
    每行代表： 一个从feature map映射回原图的点（中点）, 共九个（一行的9个坐标都相同）
    '''
    anchor_centers = tf.stack([x_centers, y_centers], 2)
    '''
    按照从上像下，从左到右的顺序展开
    '''
    anchor_centers = tf.reshape(anchor_centers, [-1, 2])

    '''
    box_sizes：
    围绕这个点生成9个anchor（每个面积下三个比例的anchor， 共9个）
    '''
    box_sizes = tf.stack([ws, hs], axis=2)
    '''
    按照从上像下，从左到右的顺序展开, 与上面的坐标对应
    '''
    box_sizes = tf.reshape(box_sizes, [-1, 2])
    '''
    anchors: anchor写成坐标点(xmin, ymin, xmax, ymax)
    按照从上像下，从左到右的顺序(第一个点生成9个anchor， 然后下一个点生成9个anchor, ....直到最后)
    '''
    anchors = tf.concat([anchor_centers - 0.5*box_sizes,
                         anchor_centers + 0.5*box_sizes], axis=1)
    return anchors

if __name__ == '__main__':
    base_anchor_size = 256
    anchor_scales = [0.5, 1., 2.0]
    anchor_ratios = [0.5, 1., 2.0]
    featuremap_height = 2
    featuremap_width = 3
    stride = 16
    anchors = make_anchors(base_anchor_size, anchor_scales, anchor_ratios,
                 featuremap_height, featuremap_width,
                 stride)

    image = np.ones(shape=[800, 800, 3], dtype=np.uint8) * 255
    image = image.astype(np.uint8)

    vis.show_result(image, anchors.numpy())
    # print(anchors)

