from flask import Flask, request
import argparse
from PIL import Image
from flask import jsonify
from face_detector import FaceDetector

app = Flask(__name__)


def build_response(status=200, code='0', data={}, message=''):
    """

    :param status: 状态码
    :param code: 返回码
    :param data: 数据
    :param message: 异常消息
    :return:
    """
    response = jsonify({"code": code, "data": data, "message": message})
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Method'] = '*'
    response.headers['Access-Control-Allow-Headers'] = '*'
    response.status_code = status
    return response


@app.route('/img/face/search', methods=['POST'])
def search_face():
    image = request.files['file']
    conf = float(request.form['conf'])
    nms = float(request.form['nms'])
    test_img = Image.open(image.stream)
    results = face_detector.detect_stream(test_img, conf, nms).tolist()
    response = build_response(200, 'success', results)
    return response


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888, help="service port", required=True)
    parser.add_argument("--gpu", type=bool, default=True, help="use gpu", required=True)
    args = parser.parse_args()

    print(args, args.gpu, args.port)
    face_detector = FaceDetector('config/yolov3-tiny.cfg', 'checkpoints/yolov3_ckpt_99.pth', gpu=args.gpu)

    print("face detector 加载完成")
    app.run(host='0.0.0.0', port=123456, debug=False, threaded=True)
