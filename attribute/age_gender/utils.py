# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import tensorflow as tf

from attribute.age_gender.data import standardize_image

RESIZE_AOI = 256
RESIZE_FINAL = 227

def _is_png(filename):
    """Determine if a file contains a PNG format image.
    Args:
    filename: string, path of the image file.
    Returns:
    boolean indicating if the image is a PNG.
    """
    return '.png' in filename

def make_multi_image_batch(filenames, coder):
    """Process a multi-image batch, each with a single-look
    Args:
    filenames: list of paths
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """

    images = []

    for filename in filenames:
        with tf.gfile.FastGFile(filename, 'rb') as f:
            image_data = f.read()
        # Convert any PNG to JPEG's for consistency.
        if _is_png(filename):
            print('Converting PNG to JPEG for %s' % filename)
            image_data = coder.png_to_jpeg(image_data)
    
        image = coder.decode_jpeg(image_data)

        crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
        image = standardize_image(crop)
        images.append(image)
    image_batch = tf.stack(images)
    return image_batch

def make_multi_crop_batch(image):
    """Process a single image file.
    Args:
    filename: string, path to an image file e.g., '/path/to/example.JPG'.
    coder: instance of ImageCoder to provide TensorFlow image coding utils.
    Returns:
    image_buffer: string, JPEG encoding of RGB image.
    """

    crops = []
    print('Running multi-cropped image')

    image = cv2.resize(image, (RESIZE_AOI, RESIZE_AOI), interpolation=cv2.INTER_CUBIC)
    print(image.shape)
    h = image.shape[0]
    w = image.shape[1]
    hl = h - RESIZE_FINAL
    wl = w - RESIZE_FINAL

    crop = tf.image.resize_images(image, (RESIZE_FINAL, RESIZE_FINAL))
    crops.append(standardize_image(crop))
    crops.append(standardize_image(tf.image.flip_left_right(crop)))

    corners = [ (0, 0), (0, wl), (hl, 0), (hl, wl), (int(hl/2), int(wl/2))]
    for corner in corners:
        ch, cw = corner
        cropped = tf.image.crop_to_bounding_box(image, ch, cw, RESIZE_FINAL, RESIZE_FINAL)
        crops.append(standardize_image(cropped))
        flipped = standardize_image(tf.image.flip_left_right(cropped))
        crops.append(standardize_image(flipped))
    image_batch = tf.stack(crops)
    return image_batch



def get_face_detection_model(model_type, model_path):
    if model_type == 'dlib':
        from attribute.age_gender.dlibdetect import FaceDetectorDlib
        return FaceDetectorDlib(model_path)
    elif model_type == 'opencv':
        from attribute.age_gender.opencvdetect import ObjectDetectorCascadeOpenCV
        return ObjectDetectorCascadeOpenCV(model_path)
    elif model_type == 'mtcnn':
        from attribute.age_gender.mtcnndetect import FaceDetectorMtcnn
        return FaceDetectorMtcnn()
