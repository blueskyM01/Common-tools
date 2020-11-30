import tensorflow as tf

'''
    两个函数的输入，输出都是一样的，即功能一样
'''

def mx_compute_overlap(anchors, gt_boxes):
    '''
    Introduction: 计算iou
    anchors:                        gt_boxes:
        [[0.0, 0.0, 1.0, 1.0],              [[0.5, 0.5, 1.0, 1.0],
         [0.2, 0.2, 1.5, 1.5],               [2.0, 1.5, 3.0, 4.0]]
         [2.2, 1.7, 2.8, 4.1]]
    输出为iou:
            [[0.25       0.        ]
             [0.14792901 0.        ]
             [0.         0.53076917]]
    从上面可以看出：是将“anchors”中的“每行”依次与“gt_boxes”中的“所有行”计算iou，
                  最终的输出iou：其“每行”对应着“anchors”中的“每行”与“bbox_target”中的“所有行”计算出的iou
                  shape=[num_anchors, num_gt_boxes]
    :param anchors: tensor, [-1, 4], 每行为：[x0, y0, x1, y1]
    :param gt_boxex:tensor, [-1, 4], 每行为：[x0, y0, x1, y1]
    :return: shape=[num_anchors, num_gt_boxes]
    '''
    anchors = tf.expand_dims(anchors, 1)
    u_x0 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 0]
    u_y0 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 1]
    u_x1 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 2]
    u_y1 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 3]

    i_x0 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 0]
    i_y0 = tf.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 1]
    i_x1 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 2]
    i_y1 = tf.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 3]

    i_w = i_x1 - i_x0
    i_h = i_y1 - i_y0
    u_w = u_x1 - u_x0
    u_h = u_y1 - u_y0

    i_w = tf.clip_by_value(i_w, 0, 1e10)
    i_h = tf.clip_by_value(i_h, 0, 1e10)
    u_w = tf.clip_by_value(u_w, 0, 1e10)
    u_h = tf.clip_by_value(u_h, 0, 1e10)

    u_arera = u_w * u_h
    i_arera = i_w * i_h
    iou = i_arera / u_arera

    return iou


def compute_overlaps(boxes1, boxes2):
    '''Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (y1, x1, y2, x2)].
    '''
    # 1. Tile boxes2 and repeate boxes1. This allows us to compare
    # every boxes1 against every boxes2 without loops.
    # TF doesn't have an equivalent to np.repeate() so simulate it
    # using tf.tile() and tf.reshape.
    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 2. Compute intersections
    b1_y1, b1_x1, b1_y2, b1_x2 = tf.split(b1, 4, axis=1)
    b2_y1, b2_x1, b2_y2, b2_x2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 3. Compute unions
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 4. Compute IoU and reshape to [boxes1, boxes2]
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


if __name__ == '__main__':
    anchors = tf.constant([[0.0, 0.0, 1.0, 1.0],
                           [0.2, 0.2, 1.5, 1.5],
                           [2.2, 1.7, 2.8, 4.1]], dtype=tf.float32)
    gt_boxes = tf.constant([[0.5, 0.5, 1.0, 1.0],
                            [2.0, 1.5, 3.0, 4.0]], dtype=tf.float32)

    iou = compute_overlaps(anchors, gt_boxes)
    print(iou)
