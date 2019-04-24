# 简介
该项目基于tensorflow框架，实现了人脸检测、人脸特征提取、人脸属性（包含年龄、性别）功能。

# 预训练模型：
opencv：

https://github.com/opencv/opencv/tree/master/data/haarcascades

dlib:

http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

facenet:

20180402-114759

https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view

# 功能模块
我们将人脸检测、人脸特征提取、人脸属性的功能进行组合。该项目支持以下几种组合：
- fd: 人脸检测；
- fr: 人脸检测和特征提取；
- fda: 人脸检测和人脸属性；
- fa: 人脸属性；
- fra: 人脸检测、特征提取以及人脸属性；
可以通过model_type这个参数来指定。

## 人脸检测
人脸检测模型包含：
- opencv
- dlib
- mtcnn

可以通过model_name这个参数来指定。
```
python demo.py --model_type 'fa' --model_name 'opencv' --file_name './data/1.jpg' --model_dir './pre_models/opencv/haarcascade_frontalface_default.xml'
```

## 人脸特征提取
需要将训练好的模型下载到pre_models中。

```
python demo.py --model_type 'fr' --model_name 'facenet' --file_name './data/2.jpg' --model_dir './pre_models/dlib/shape_predictor_68_face_landmarks.dat'
```

## 人脸属性
目前支持age以及gender，表情的代码在陆续开发中。

```
python demo.py --model_type 'fa' --model_name 'mtcnn' --file_name './data/2.jpg'
```

