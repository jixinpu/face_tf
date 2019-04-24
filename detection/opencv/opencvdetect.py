# -*- coding: utf-8 -*-

import cv2
import numpy as np

FACE_PAD = 50

class ObjectDetectorCascadeOpenCV():
    def __init__(self, model_name, basename='frontal-face', tgtdir='.', min_height_dec=20, min_width_dec=20,
                 min_height_thresh=50, min_width_thresh=50):
        self.min_height_dec = min_height_dec
        self.min_width_dec = min_width_dec
        self.min_height_thresh = min_height_thresh
        self.min_width_thresh = min_width_thresh
        self.tgtdir = tgtdir
        self.basename = basename
        self.face_cascade = cv2.CascadeClassifier(model_name)

    def run(self, img):
        min_h = int(max(img.shape[0] / self.min_height_dec, self.min_height_thresh))
        min_w = int(max(img.shape[1] / self.min_width_dec, self.min_width_thresh))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, minNeighbors=5, minSize=(min_h, min_w))

        face_info = {}
        for index, rect in enumerate(faces):
            bbox = {}
            bbox['left'] = rect[0]
            bbox['top'] = rect[1]
            bbox['right'] = rect[0] + rect[2]
            bbox['bottom'] = rect[1] + rect[3]
            face_info[index] = bbox

        return face_info

