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
            bbox = {}
            bbox['top'] = int(rect.top())
            bbox['bottom'] = int(rect.bottom())
            bbox['left'] = int(rect.left())
            bbox['right'] = int(rect.right())
            face_info[index] = bbox

        return face_info
