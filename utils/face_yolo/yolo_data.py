"""
获取yolo格式的数据
"""
from __future__ import division
import os
import cv2

image_root = "E:\\datasets\\face\\WIDER_val\\images"
label_root = "E:\\datasets\\face\\wider_face_split\\wider_face_val_bbx_gt.txt"
yolo_label_root = "E:\\datasets\\face\\WIDER_val\\labels"

if not os.path.exists(yolo_label_root):
    os.makedirs(yolo_label_root)


label_file = open(label_root, encoding='utf-8')

while label_file:
    line = label_file.readline().strip('\n')

    image_path = os.path.join(image_root, line)
    label_path = image_path.replace("images", "labels").replace(".png", ".txt").replace(".jpg", ".txt")

    paths = line.split('/')
    child_path = os.path.join(yolo_label_root, paths[0])
    if not os.path.exists(child_path):
        os.makedirs(child_path)

    img = cv2.imread(image_path)
    img_w = img.shape[1]
    img_h = img.shape[0]

    face_num = int(label_file.readline().strip('\n'))
    yolo_file = open(label_path, 'w+', encoding='utf-8')

    if face_num == 0:
        label_file.readline()
        yolo_file.close()
        continue
    for i in range(face_num):
        face_info = label_file.readline().strip('\n').split(' ')
        x = int(face_info[0])
        y = int(face_info[1])
        w = int(face_info[2])
        h = int(face_info[3])
        p1 = (x, y)
        p2 = (x + w, y + h)
        yolo_x = x + int(w / 2)
        yolo_y = y + int(h / 2)
        yolo_w = w
        yolo_h = h
        cv2.rectangle(img, p1, p2, (0, 255, 0), 1)
        yolo_file.write('{} {} {} {} {}\n'.format(0, yolo_x / img_w, yolo_y / img_h, yolo_w / img_w, yolo_h / img_h))
    yolo_file.close()


label_file.close()
