# -*- coding: utf-8 -*-

import cv2

def face_recognition(image_name, model_name, model_dir):
    img = cv2.imread(image_name)
    if model_name == 'facenet':
        from mtcnn_facenet import facenet_recognition
        recognition_info = facenet_recognition.feature_extraction(img, model_dir)
        return recognition_info