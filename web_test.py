import requests
import time
import json
import cv2


def find_face(frame, conf=0.99, nms=0.1):
    bytes = cv2.imencode(".jpg", frame)[1].tobytes()
    files = {'file': bytes}

    upload_data = {"conf": conf, "nms": nms}
    start = time.time()
    res = requests.post('http://192.168.106.233:57920/img/face/search', upload_data, files=files)
    print(time.time() - start)
    return json.loads(res.content)['data']


capture = cv2.VideoCapture(
    'C:\\Users\\jky\\Documents\\Tencent Files\\136403122\\FileRecv\\10.110.47.181_IPC_main_20190612191049.dav')

num = 0
while True:
    ret, frame = capture.read()

    num += 1

    if num % 5 == 0:
        frame = cv2.resize(frame, (960, 640))
        results = find_face(frame, conf=0.9, nms=0.1)

        for result in results:
            cv2.rectangle(frame, (int(result[0]), int(result[1])), (int(result[2]), int(result[3])), (0, 255, 0), 1)

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1)
        if key == ord('r'):
            cv2.waitKey()
        elif key == ord('q'):
            break
