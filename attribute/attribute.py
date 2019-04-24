# -*- coding: utf-8 -*-

import cv2

from age_gender import age_gender
from age_gender.utils import *

attribute_list = ['age', 'gender']

def face_attribute(image_name, face_detection_type, face_detection_model):
    face_info = {}
    img = cv2.imread(image_name)
    if face_detection_model:
        print('Using face detector (%s) %s' % (face_detection_type, face_detection_model))
        face_detect = get_face_detection_model(face_detection_type, face_detection_model)
        face_info = face_detect.run(img)

    print("%d faces have been found." %len(face_info))
    some_face_attribute_info = {}
    if len(face_info) > 0:
        for index in face_info:
            one_face_attribute_info = {}
            for i in range(len(attribute_list)):
                if attribute_list[i] == 'age':
                    model_type = 'inception'
                    device_id = '/cpu:0'
                    model_dir = './pre_models/age_model'
                    result, prob = age_gender.classification(face_info[index], attribute_list[i], model_type, device_id, model_dir)
                    one_face_attribute_info[attribute_list[i]] = str(result) + '--' + str(prob)

                if attribute_list[i] == 'gender':
                    model_type = 'inception'
                    device_id = '/cpu:0'
                    model_dir = './pre_models/gender_model'
                    result, prob = age_gender.classification(face_info[index], attribute_list[i], model_type, device_id, model_dir)
                    one_face_attribute_info[attribute_list[i]] = str(result) + '--' + str(prob)
            some_face_attribute_info[str(index)] = one_face_attribute_info

    print(some_face_attribute_info)