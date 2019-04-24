# -*- coding: utf-8 -*-

import dlib
import cv2
import base64
import numpy as np

FACE_PAD = 50

class FaceDetectorDlib():
    def __init__(self, model_name, basename='frontal-face', tgtdir='.'):
        self.tgtdir = tgtdir
        self.basename = basename
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_name)

    def run(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 1)
        face_info = {}
        for (index, rect) in enumerate(faces):
            x = rect.left()
            y = rect.top()
            w = rect.right() - x
            h = rect.bottom() - y

            # 生成新的空白图像
            img_blank = np.zeros((h, w, 3), np.uint8)

            # 将人脸写入空白图像
            for i in range(h):
                for j in range(w):
                    img_blank[i][j] = img[y + i][x + j]
            face_info[index] = img_blank

        return face_info
