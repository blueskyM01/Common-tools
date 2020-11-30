import cv2
import numpy as np

def show_result(image):
    cv2.imshow('show result', image)
    cv2.waitKey(0)

if __name__ == '__main__':
    image = np.ones(shape=[255,255, 3], dtype=np.uint8)
    print('1234')
    show_result(image)
