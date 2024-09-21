from reader import Reader
from model import CRNN
import torch
import os
import PIL.Image as Image
import numpy as np
import torchvision.transforms.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"
checkpoint_path = "checkpoint/ckpt-1719416455.pth"
file_path = "sample_img/3952.jpg"

IMAGE_SHAPE_C = 3
IMAGE_SHAPE_H = 30
IMAGE_SHAPE_W = 70


# trans = T.Compose(
#     [
#         T.Resize((IMAGE_SHAPE_H, IMAGE_SHAPE_W)),
#         T.ToTensor(),
#         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#     ]
# )

def test():
    # 读取测试图片并进行分类
    net = CRNN()
    net = net.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint["stat"])
    net.eval()
    with torch.no_grad():
        img = Image.open(file_path)
        img = img.resize((IMAGE_SHAPE_W, IMAGE_SHAPE_H))
        # # 转为Numpy的array格式并整体除以255进行归一化
        img = (
            np.array(img, dtype="float32").reshape(
                (IMAGE_SHAPE_C, IMAGE_SHAPE_H, IMAGE_SHAPE_W)
            )
            / 255
        )

        # img = trans(img).to(device)
        img = torch.tensor(img).to(device)

        # 进行分类
        
        img = torch.unsqueeze(img, 0)
        out = net(img).log_softmax(2)
        pre_label = torch.argmax(out, -1).cpu().numpy()
        pre_label = pre_label[pre_label !=10]
        print(f"pre_label: {pre_label}")


if __name__ == "__main__":
    test()
