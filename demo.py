# -*- coding: utf-8 -*-

import tensorflow as tf

from attribute import attribute
from detection import detection
from recognition import recognition

tf.app.flags.DEFINE_string('model_type', '',
                           'Model type (fd | fr | fa | fra)')
tf.app.flags.DEFINE_string('model_name', '',
                           'Model name')

tf.app.flags.DEFINE_string('file_name', '',
                           'File name')

tf.app.flags.DEFINE_string('model_dir', '',
                           'Model dir')

FLAGS = tf.app.flags.FLAGS

def main():
    if FLAGS.model_type == 'fd':
        print('face detection')
        detection_info = detection.face_detector(FLAGS.file_name, FLAGS.model_name, FLAGS.model_dir)
        print(detection_info)
    if FLAGS.model_type == 'fr':
        print('face recognition')
        feature_info = recognition.face_recognition(FLAGS.file_name, FLAGS.model_name, FLAGS.model_dir)
        print(feature_info)
    if FLAGS.model_type == 'fa':
        print('face attribute')
        attribute.face_attribute(FLAGS.file_name, FLAGS.model_name, FLAGS.model_dir)
    if FLAGS.model_type == 'fra':
        print('face recogition and attribute')

if __name__ == "__main__":
    main()