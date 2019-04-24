# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import base64

from detection.mtcnn.align import detect_face

# minimum size of face
minsize = 20
# three steps's threshold
threshold = [0.6, 0.7, 0.7]
# scale factor
factor = 0.709

class FaceDetectorMtcnn():
    def __init__(self):
        print('Creating networks and loading parameters')
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, None)
        self.pnet = pnet
        self.rnet = rnet
        self.onet = onet

    def run(self, img):
        img, det, crop_image, confidence, j = mtcnn(img, self.pnet, self.rnet, self.onet)
        face_info = {}
        for i in range(len(det)):
            y = int(det[i, 1])
            h = int(det[i, 3]) - y
            x = int(det[i, 0])
            w = int(det[i, 2]) - x

            # 生成新的空白图像
            img_blank = np.zeros((h, w, 3), np.uint8)

            # 将人脸写入空白图像
            for m in range(h):
                for n in range(w):
                    img_blank[m][n] = img[y + m][x + n]
            face_info[i] = img_blank
        return face_info

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y

def mtcnn(img, pnet, rnet, onet):
    """使用mtcnn进行人脸检测"""
    #img = base64_to_image(base64_data)
    image_size = 160
    # 获取图片的shape
    img_size = np.asarray(img.shape)[0:2]
    # 返回边界框数组（参数分别是输入图片 脸部最小尺寸 三个网络 阈值）
    bounding_boxes, _ = detect_face.detect_face(img, minsize, pnet, rnet, onet, threshold, factor)

    if len(bounding_boxes) < 1:
        return 0,0,0,0,0
    else:
        crop = []
        confidence = []
        det = bounding_boxes

        det[:, 0] = np.maximum(det[:, 0], 0)
        det[:, 1] = np.maximum(det[:, 1], 0)
        det[:, 2] = np.minimum(det[:, 2], img_size[1])
        det[:, 3] = np.minimum(det[:, 3], img_size[0])

        for i in range(len(bounding_boxes)):
            temp_crop = img[int(det[i, 1]):int(det[i, 3]), int(det[i, 0]):int(det[i, 2]), :]
            aligned = misc.imresize(temp_crop, (image_size, image_size), interp='bilinear')
            prewhitened = prewhiten(aligned)
            crop.append(prewhitened)
            confidence.append(det[i, 4])
        crop_image = np.stack(crop)

        return img, det, crop_image, confidence, 1


