# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
from scipy import misc
import tensorflow as tf
import numpy as np
import base64
from recognition.mtcnn_facenet import facenet
from recognition.mtcnn_facenet.align import detect_face

# minimum size of face
minsize = 20
# three steps's threshold
threshold = [0.6, 0.7, 0.7]
# scale factor
factor = 0.709

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

def mtcnn_facenet_embeding(img, pnet, rnet, onet):
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
            prewhitened = facenet.prewhiten(aligned)
            crop.append(prewhitened)
            confidence.append(det[i, 4])
        crop_image = np.stack(crop)

        return img, det, crop_image, confidence, 1

def feature_extraction(img, model_dir):
    print('Creating networks and loading parameters')
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, None)

    with tf.Graph().as_default():
        with tf.Session() as sess:
    	    facenet.load_model(model_dir)
            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

            # -------建立路由----------
            img, det, crop_image, confidence, j = mtcnn_facenet_embeding(img, pnet, rnet, onet)

            if j:
                feed_dict = {images_placeholder: crop_image, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
                faces_info = []
                for i in range(len(emb)):
                    box_info = {}
                    face_item = {}
                    box_info['top']    = int(det[i, 1])
                    box_info['bottom'] = int(det[i, 3])
                    box_info['left']   = int(det[i, 0])
                    box_info['right']  = int(det[i, 2])
                    box_info['height'] = img.shape[0]
                    box_info['width']  = img.shape[1]

                    face_base64 = get_face_base64(box_info, img)

                    box_info['confidence'] = confidence[i]
                    face_item['feature'] = emb[i]
                    face_item['box'] = box_info
                    face_item['pic'] = face_base64
                    faces_info.append(face_item)

    return faces_info
