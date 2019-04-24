# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from attribute.age_gender.model import select_model, get_checkpoint
from attribute.age_gender.utils import *

RESIZE_FINAL = 227
GENDER_LIST =['M','F']
AGE_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
MAX_BATCH_SZ = 128

def classify_one_multi_crop(sess, label_list, softmax_output, images, face_img):
    image_batch = make_multi_crop_batch(face_img)
    with tf.Session() as sess1:
        image_batch1 = sess1.run(image_batch)
        batch_results = sess.run(softmax_output, feed_dict={images:image_batch1})
        print(batch_results)
        output = batch_results[0]
        batch_sz = batch_results.shape[0]
        
        for i in range(1, batch_sz):
            output = output + batch_results[i]

        output /= batch_sz
        best = np.argmax(output)
        best_choice = (label_list[best], output[best])

        return label_list[best], output[best]

def classification(face_info, class_type, model_type, device_id, model_dir):
    config = tf.ConfigProto(allow_soft_placement=True)
    # 清除每次运行的图
    tf.reset_default_graph()
    with tf.Session(config=config) as sess:
        label_list = AGE_LIST if class_type == 'age' else GENDER_LIST
        nlabels = len(label_list)

        print('Executing on %s' % device_id)
        model_fn = select_model(model_type)

        with tf.device(device_id):
            
            images = tf.placeholder(tf.float32, [None, RESIZE_FINAL, RESIZE_FINAL, 3])
            logits = model_fn(nlabels, images, 1, False)
            init = tf.global_variables_initializer()
        
            checkpoint_path = '%s' % (model_dir)

            model_checkpoint_path, global_step = get_checkpoint(checkpoint_path)

            saver = tf.train.Saver()
            saver.restore(sess, model_checkpoint_path)
                        
            softmax_output = tf.nn.softmax(logits)

            out, result = classify_one_multi_crop(sess, label_list, softmax_output, images, face_info)

            return out, result
