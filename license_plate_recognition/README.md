# 车牌检测与识别：Web APP 部署

## 简介

本项目基于 CRNN 模型搭建车牌自动识别系统。在系统中上传车牌图片，系统自动识别车牌号并显示。

### 目录结构说明

```
.
├── README.md                   # 项目说明文档
└── ocrcar                      # 系统根目录
    ├── infer.py                # 模型推理程序
    ├── models                  # Yolo检测器目录
    │   ├── common.py           # 工具方法
    │   ├── detect_plate.py     # 检测器推理程序
    │   ├── general.py          # 工具方法
    │   ├── save_models         # 存放训练好的模型参数
    │   └── yolo.py             # Yolo检测模型
    ├── recognizer              # 车牌识别模型目录
    │   ├── model.py            # 车牌识别模型
    │   └── save_models         # 存放训练好的模型参数
    ├── requirements.txt        # 项目相关依赖
    ├── run.sh                  # 系统启动脚本
    ├── server.py               # 服务器后端程序
    └── templates               # 存放前端页面
        ├── apphome.html        # 手机端页面
        ├── error.html          # 错误页面
        └── home.html           # PC端页面
```

## 系统架构

`server.py`中将通过 Flask 启动一个 Web 服务。这个 FLASK 服务器前接用户来自浏览器的请求，后接用于推理图片结果的 infer.py。架构如下所示：

```
┌───────┐           ┌───────┐        ┌───────┐
│       │    AJAX   │       │        │       │
│       ├───────────►       ├────────►       │
│ USER  │           │ FLASK │        │ INFER │
│       ◄───────────┤       ◄────────┤       │
│       │   OUTPUT  │       │        │       │
└───────┘           └───────┘        └───────┘
```

### 前端

前端实现代码在`templates/home.html`中，它提供了一个`uploader`用于用户拍照/上传图片：

```
<van-uploader id="image" name="image" :after-read="afterRead" :max-count="1" />
```

当用户拍照/上传图片后，通过`axios`发送 POST 请求给后端，并且将后端返回的预测结果更新到前端页面上：

```
axios.post("", formData).then((res) => {
                console.log('Upload success');
                vue_obj.image_url = res.data.data.image_url;
                vue_obj.prediction = res.data.data.prediction;
                vant.Toast(res.data.data.prediction);
        });
```

因为用户拍摄的照片往往较大，所以上传照片前，会预先对照片进行压缩：

```
new Compressor(file.file, {
              //...
            });
```

### 后端

在 server.py 中，注册了路由`/v1/index`：

```
@app.route(f"{base_dir}/v1/index", methods=["GET", "POST"])
#...
```

接口`/v1/index`返回状态码及其含义：

| 状态码 |       英文名称       |     中文描述     |
| :----: | :------------------: | :--------------: |
|   0    |       SUCCESS        | 车牌检测识别成功 |
|   -1   |         FAIL         | 车牌检测识别失败 |
|  4001  | FILE_SELECTION_ERROR |   文件选择错误   |
|  4002  |  UPLOAD_FILE_EMPTY   |   上传文件为空   |
|  5001  |  BACKEND_EXCEPTION   |     后台异常     |

当接受 GET 请求时，返回主页面（home.html）。当接受 POST 请求时，则做两件事情：

- 保存图片（方便之后被前端引用显示）；
- 调用模型对图片进行预测并返回预测结果；

保存图片相关代码：

```
filename = generate_filenames(image_file.filename)
filePath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
image_file.save(filePath)
```

对图片进行预测相关代码：

```
def predict(filename):
    original_image_url = url_for("images", filename=filename)
    original_image_file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    try:
        out = model.inference(original_image_file_path)
        json_obj = {"code": ResponseCode.SUCCESS, "data": {
                    "image_url": original_image_url, "prediction": out}, "msg": ResponseMessage.SUCCESS}
        return json.dumps(json_obj)
    except:
        json_obj = {"code": ResponseCode.SUCCESS, "data": {
                    "image_url": original_image_url, "prediction": "无车牌"}, "msg": ResponseMessage.SUCCESS}
        return json.dumps(json_obj)
```

### 推理

后端推理使用 PyTorch，通过读取已经训练好的 CRNN 模型文件 `recognizer/save_models/car_plate_crnn.pt` 初始化模型。

```
def loadModel(self):
    self.model = CRNN(self.nhidden, self.nclass).eval()
    self.model.to(self.DEVICE)
    self.model.load_state_dict(torch.load(self.model_path))
```

输入待识别的图片，模型进行推理并输出识别结果：

```
def inference(self,imgPath=""):
    image = self.readImg(imgPath)
    output = self.model(image).cpu()
    output = torch.squeeze(output)
    _, indexs = output.max(1)
    output_label = self.parseTest(indexs)
    return  output_label
```

## 使用指南

“Fork” 本项目，并填写相关信息。（可[下载](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/%E8%B5%A3ZDB16Q.jpg)测试车牌）

### 1、项目部署

点击**部署**按钮，选择项目文件 ocrcar ：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/deploy1.png)

填写基本信息：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/deploy2.png)

选择运行环境，并设置启动命令`cd /workspace/ocrcar && ./run.sh`，端口号设置为 5000：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/deploy3.png)

### 2、创建容器

点击**运行**按钮，选择服务器型号：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/run1.png)

点击**确定**按钮，容器开始创建：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/create.png)

创建完成后显示如下：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/show.png)

### 3、系统运行

点击**测试**按钮进入 系统前端页面：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/web1.png)

上传需要识别的图片，系统返回识别的结果：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/web2.png)

点击“手机端”，手机扫码进入：

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220218-leitao-voiceclone/app.png)

![image](https://oneflow-public.oss-cn-beijing.aliyuncs.com/OneCloud/img/20220117-leitao-ocr_car/res.png)
