from models import *
from torchvision import transforms
from PIL import Image
from utils.utils import *
import torch
import cv2
import time


def resize(image, size):
    image = F.interpolate(image.unsqueeze(0), size=size, mode="nearest").squeeze(0)
    return image


def pad_to_square(img, pad_value):
    c, h, w = img.shape
    dim_diff = np.abs(h - w)
    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    # Determine padding
    pad = (0, 0, pad1, pad2) if h <= w else (pad1, pad2, 0, 0)
    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img, pad


class FaceDetector(object):

    def __init__(self, config_file, weight_file, img_size=416, gpu=False):
        self.model = Darknet(config_file, img_size=img_size)
        self.img_size = img_size
        print(gpu)
        if gpu:
            self.device = 'cuda:1'
            self.model.to(self.device)
            self.model.load_state_dict(torch.load(weight_file))

            print('use gpu')
        else:
            self.model.load_state_dict(torch.load(weight_file, map_location='cpu'))
            print('use cpu')

        self.model.eval()
        self.to_tensor = transforms.ToTensor()

    def detect(self, img_path, conf=0.1, nms=0.1):
        stream = Image.open(img_path)
        return self.detect_stream(stream, conf, nms)

    def detect_stream(self, stream, conf=0.5, nms=0.3):

        origin_type = np.array(stream).shape[:2]

        img = self.to_tensor(stream)

        img, _ = pad_to_square(img, 0)

        img = resize(img, self.img_size)

        img = torch.unsqueeze(img, 0)

        with torch.no_grad():
            detections = self.model(img.to(self.device), device=self.device)

            detections = non_max_suppression(detections.cpu(), conf, nms)
            if detections:
                detections = rescale_boxes(detections[0], self.img_size, origin_type)

            return detections.detach().data.numpy()

    def detect_cv(self, array, conf=0.5, nms=0.3):
        image = Image.fromarray(cv2.cvtColor(array, cv2.COLOR_BGR2RGB))
        return self.detect_stream(image, conf, nms)

# image_path = 'data/face/images/yxd_4.jpg'
# results = face_detector.detect(image_path)
#
# print(results)

# capture = cv2.VideoCapture(
#     'C:\\Users\\jky\\Documents\\Tencent Files\\136403122\\FileRecv\\10.110.47.181_IPC_main_20190612191049.dav')

# num = 0
# while True:
#     ret, frame = capture.read()
#
#     num += 1
#
#     if num % 30 == 0:
#         frame = cv2.resize(frame, (960, 640))
#
#         results = face_detector.detect_cv(frame, conf=0.99, nms=0.1)
#
#         for result in results:
#             cv2.rectangle(frame, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 255, 0), 1)
#
#         cv2.imshow('frame', frame)
#         key = cv2.waitKey(1)
#         if key == ord('r'):
#             cv2.waitKey()
