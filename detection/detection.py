# -*- coding: utf-8 -*-

import cv2

def display_face_detector(img, face_info):
    for index in face_info:
        left = face_info[index]['left']
        top = face_info[index]['top']
        right = face_info[index]['right']
        bottom = face_info[index]['bottom']
        cv2.rectangle(img, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.imshow('facedetector', img)

    k = cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_face_detection_model(model_type, model_path):
    if model_type == 'dlib':
        from dlib.dlibdetect import FaceDetectorDlib
        return FaceDetectorDlib(model_path)
    elif model_type == 'opencv':
        from opencv.opencvdetect import ObjectDetectorCascadeOpenCV
        return ObjectDetectorCascadeOpenCV(model_path)
    elif model_type == 'mtcnn':
        from mtcnn.mtcnndetect import FaceDetectorMtcnn
        return FaceDetectorMtcnn(model_path)

def face_detector(image_name, model_name, model_dir):
    img = cv2.imread(image_name)
    face_detect = get_face_detection_model(model_name, model_dir)

    # ---------路由-------------------
    face_info = face_detect.run(img)
    print(face_info)

    display_face_detector(img, face_info)