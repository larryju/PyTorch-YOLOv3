"""
获取yolo格式的数据
"""
from __future__ import division
import os


image_root = "E:\\datasets\\face\\WIDER_val\\images"

train_txt = open('valid.txt', encoding='utf-8', mode='a+')

prefix = image_root

for child_path in os.listdir(image_root):
    for img in os.listdir(os.path.join(image_root, child_path)):
        line = os.path.join(image_root, child_path, img)
        train_txt.write(line + '\n')

train_txt.close()
