
import torch
from torch import nn
import os
import json
import torchvision.models as models

from PIL import Image
from torchvision import transforms

# os.chdir("/content/drive/MyDrive/traffic-signs-classification/")

cwd = os.getcwd()
test_dir = os.path.join(cwd, 'data', 'test_dataset')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
checkpoint_path = os.path.join(cwd, 'checkpoint', 'ckpt.pth')

result = {
    "annotations": []
}
result_file = os.path.join(cwd, 'result.json')

test_data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 224
    transforms.RandomHorizontalFlip(),  # 默认概率0.5
    transforms.ToTensor(),
    # 图像标准化处理
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


class Dawei(nn.Module):
    # 搭建神经网络
    def __init__(self, num_class=10):
        super(Dawei, self).__init__()
        vgg16_net = models.vgg16_bn(pretrained=False)
        self.features = vgg16_net.features
        self.avgpool = vgg16_net.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(1024, 256),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Linear(64, num_class),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def test():
    # 读取测试图片并进行分类
    net = Dawei()
    net = net.to(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
    net.load_state_dict(checkpoint['net'])
    net.eval()
    index = 0
    with torch.no_grad():
        for image_name in os.listdir(test_dir):
            image = Image.open(os.path.join(test_dir, image_name))
            image = image.convert('RGB')
            index += 1
            print('{}:{}'.format(index, image))
            image = test_data_transforms(image)
            image = image.to(device)
            image = torch.reshape(image, (1, 3, 224, 224))
            # 进行分类
            out = net(image)
            pre_label = torch.argmax(out, 1).item()
            item = {
                "filename": "{}/{}".format("test_dataset", image_name),
                "label": pre_label
            }
            print(item)
            result["annotations"].append(item)

    # dump
    with open(result_file, 'a+') as file:
        result_str = json.dumps(result)
        result_json = json.loads(result_str)
        json.dump(result_json, file)


if __name__ == '__main__':
    test()
