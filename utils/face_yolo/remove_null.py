"""
去除为空的图片
"""

import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="E:\\datasets\\face\\WIDER_train\\images",
                        help="root of image")
    parser.add_argument("--out_name", type=str, default='train.txt', help="images txt")

    opt = parser.parse_args()

    out_name = opt.out_name
    image_root = opt.image_path

    # train_txt = open(out_name, encoding='utf-8', mode='a+')

    for child_path in os.listdir(image_root):
        for img in os.listdir(os.path.join(image_root, child_path)):
            line = os.path.join(image_root, child_path, img)
            # train_txt.write(line + '\n')
            line_label = line.replace('images', 'labels').replace('jpg', 'txt')
            if os.path.getsize(line_label) == 0:
                print(line_label)
