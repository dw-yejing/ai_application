import os
import random
import qrcode
import requests

from PIL import Image
from io import BytesIO
from infer import CarOCR
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, url_for, send_file, jsonify


app = Flask(__name__)
app.config["SECRET_KEY"] = "12345"
app.config["UPLOAD_FOLDER"] = "./temp"
base_dir = os.environ.get("BASE_DIR", "")

model = CarOCR(256)


class ResponseCode(object):
    SUCCESS = 0  # 车牌检测识别成功
    FAIL = -1  # 车牌检测识别失败
    FILE_SELECTION_ERROR = 4001  # 文件选择错误
    UPLOAD_FILE_EMPTY = 4002  # 上传文件为空
    BACKEND_EXCEPTION = 5001  # 后台异常


class ResponseMessage(object):
    SUCCESS = "车牌检测识别成功"
    FAIL = "车牌检测识别失败"
    FILE_SELECTION_ERROR = "文件选择错误"
    UPLOAD_FILE_EMPTY = "上传文件为空"
    BACKEND_EXCEPTION = "后台异常"


@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
def home():
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    if request.method == "GET":
        code_url = generate_code()
        return render_template("home.html", base_dir=base_dir, param_value=code_url)
    if request.method == "POST":
        if "image" not in request.files:
            json_obj = {"code": ResponseCode.FILE_SELECTION_ERROR, "data": {
                "image_url": "", "prediction": ""}, "msg": ResponseMessage.FILE_SELECTION_ERROR}
            return jsonify(json_obj)
        image_file = request.files["image"]

        if image_file.filename == "":
            json_obj = {"code": ResponseCode.UPLOAD_FILE_EMPTY, "data": {
                "image_url": "", "prediction": ""}, "msg": ResponseMessage.UPLOAD_FILE_EMPTY}
            return jsonify(json_obj)

        if image_file and is_allowed_file(image_file.filename):
            try:
                filename = generate_filenames(image_file.filename)
                filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image_file.save(filePath)
                return predict(filename)
            except Exception:
                json_obj = {"code": ResponseCode.BACKEND_EXCEPTION, "data": {
                    "image_url": "", "prediction": ""}, "msg": ResponseMessage.BACKEND_EXCEPTION}
                return jsonify(json_obj)


@app.route(f"{base_dir}/v1/app", methods=["GET", "POST"])
def apphome():
    if not os.path.exists(app.config["UPLOAD_FOLDER"]):
        os.makedirs(app.config["UPLOAD_FOLDER"])
    if request.method == "GET":
        return render_template("apphome.html", base_dir=base_dir)
    if request.method == "POST":
        if "image" not in request.files:
            json_obj = {"code": ResponseCode.FILE_SELECTION_ERROR, "data": {
                "image_url": "", "prediction": ""}, "msg": ResponseMessage.FILE_SELECTION_ERROR}
            return jsonify(json_obj)
        image_file = request.files["image"]

        if image_file.filename == "":
            json_obj = {"code": ResponseCode.UPLOAD_FILE_EMPTY, "data": {
                "image_url": "", "prediction": ""}, "msg": ResponseMessage.UPLOAD_FILE_EMPTY}
            return jsonify(json_obj)

        if image_file and is_allowed_file(image_file.filename):
            try:
                filename = generate_filenames(image_file.filename)
                filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                image_file.save(filePath)
                return predict(filename)
            except Exception:
                json_obj = {"code": ResponseCode.BACKEND_EXCEPTION, "data": {
                    "image_url": "", "prediction": ""}, "msg": ResponseMessage.BACKEND_EXCEPTION}
                return jsonify(json_obj)


@app.route(f"{base_dir}/images/<filename>", methods=["GET"])
def images(filename):
    return send_file(os.path.join(app.config["UPLOAD_FOLDER"], filename))


@app.errorhandler(500)
def server_error(error):
    return render_template("error.html"), 500


def is_allowed_file(filename):
    VALID_EXTENSIONS = ["png", "jpg", "jpeg"]
    is_valid_ext = filename.rsplit(".", 1)[1].lower() in VALID_EXTENSIONS
    return "." in filename and is_valid_ext


def generate_code():
    qr = qrcode.QRCode(
        version=2,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=10,
        border=1
    )
    qr.add_data(
        "https://gateway.platform.oneflow.cloud{}/v1/app".format(base_dir))
    qr.make(fit=True)
    img = qr.make_image()

    img = img.convert("RGBA")
    img_w, img_h = img.size
    factor = 3
    size_w = int(img_w / factor)
    size_h = int(img_h / factor)
    of_image = "https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/of.png"
    yzmdata = requests.get(of_image)
    tempIm = BytesIO(yzmdata.content)
    icon = Image.open(tempIm)
    icon_w, icon_h = icon.size
    if icon_w > size_w:
        icon_w = size_w
    if icon_h > size_h:
        icon_h = size_h
    icon = icon.resize((icon_w, icon_h), Image.ANTIALIAS)

    w = int((img_w - icon_w) / 2)
    h = int((img_h - icon_h) / 2)
    img.paste(icon, (w, h), icon)
    ercode_image = os.path.join(
        app.config["UPLOAD_FOLDER"], 'qrcode.png')
    img.save(ercode_image)
    return url_for("images", filename="qrcode.png")


def generate_filenames(filename):
    LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    ext = filename.split(".")[-1]
    random_indexes = [random.randint(0, len(LETTERS) - 1) for _ in range(10)]
    random_chars = "".join([LETTERS[index] for index in random_indexes])
    new_name = "{name}.{extension}".format(name=random_chars, extension=ext)
    return secure_filename(new_name)


def predict(filename):
    print('filename:'+filename)
    original_image_url = url_for("images", filename=filename)
    original_image_file_path = os.path.join(
        app.config["UPLOAD_FOLDER"], filename)
    try:
        out = model.inference(original_image_file_path)
        json_obj = {"code": ResponseCode.SUCCESS, "data": {
                    "image_url": original_image_url, "prediction": out}, "msg": ResponseMessage.SUCCESS}
        return jsonify(json_obj)
    except:
        json_obj = {"code": ResponseCode.SUCCESS, "data": {
                    "image_url": original_image_url, "prediction": "无车牌"}, "msg": ResponseMessage.SUCCESS}
        return jsonify(json_obj)


if __name__ == "__main__":
    print("server is running .....")
    app.run("0.0.0.0")
