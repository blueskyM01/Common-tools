import cv2
import numpy as np

def show_result(image, anchors):
    img = image.copy()

    for b in anchors:
        xmin = int(b[0])
        ymin = int(b[1])
        xmax = int(b[2])
        ymax = int(b[3])
        center_x = int((b[0]+b[2]) / 2)
        center_y = int((b[1]+b[3]) / 2)
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), ((0, 0, 255)), thickness=2)
        cv2.circle(img, (center_x, center_y), radius=2, color=[0,0,0], thickness=-1)
    cv2.imshow('show result', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    image = np.ones(shape=[800,800, 3], dtype=np.uint8) * 255
    image = image.astype(np.uint8)
    print('1234')
    show_result(image)
