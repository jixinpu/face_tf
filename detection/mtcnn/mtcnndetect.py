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
    def __init__(self, model_path):
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
            bbox = {}
            face_item = {}
            bbox['top'] = int(det[i, 1])
            bbox['bottom'] = int(det[i, 3])
            bbox['left'] = int(det[i, 0])
            bbox['right'] = int(det[i, 2])
            face_info[i] = bbox

        return face_info

def get_face_base64(box_info, img_rgb):
    height = box_info['bottom'] - box_info['top']
    width  = box_info['right'] - box_info['left']
    img_blank = np.zeros((height, width, 3), np.uint8)
    for i in range(height):
        for j in range(width):
            img_blank[i][j] = img_rgb[box_info['top'] + i][box_info['left'] + j]
    image = cv2.imencode('.jpg',img_blank)[1]
    image_code = str(base64.b64encode(image))

    return image_code

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