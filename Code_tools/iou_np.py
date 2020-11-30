import numpy as np


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
    anchors = np.expand_dims(anchors, 1)
    u_x0 = np.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 0]
    u_y0 = np.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 1]
    u_x1 = np.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 2]
    u_y1 = np.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 3]

    i_x0 = np.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 0]
    i_y0 = np.where(anchors > gt_boxes, anchors, gt_boxes)[:, :, 1]
    i_x1 = np.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 2]
    i_y1 = np.where(anchors < gt_boxes, anchors, gt_boxes)[:, :, 3]

    i_w = i_x1 - i_x0
    i_h = i_y1 - i_y0
    u_w = u_x1 - u_x0
    u_h = u_y1 - u_y0

    i_w = np.clip(i_w, 0, 1e10)
    i_h = np.clip(i_h, 0, 1e10)
    u_w = np.clip(u_w, 0, 1e10)
    u_h = np.clip(u_h, 0, 1e10)

    u_arera = u_w * u_h
    i_arera = i_w * i_h
    iou = i_arera / u_arera

    return iou


if __name__ == '__main__':
    anchors = np.array([[0.0, 0.0, 1.0, 1.0],
                        [0.2, 0.2, 1.5, 1.5],
                        [2.2, 1.7, 2.8, 4.1]], dtype=np.float32)
    gt_boxes = np.array([[0.5, 0.5, 1.0, 1.0],
                         [2.0, 1.5, 3.0, 4.0]], dtype=np.float32)
    iou = mx_compute_overlap(anchors, gt_boxes)
    print(iou)
