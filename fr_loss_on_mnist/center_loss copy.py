import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.center = nn.Parameter(10 * torch.randn(10, 2))
        self.lamda = 0.2
        self.weight = nn.Parameter(torch.Tensor(2, 10))  # (input,output)
        nn.init.xavier_uniform_(self.weight)

    def forward(self, feature, label):
        batch_size = label.size()[0]
        nCenter = self.center.index_select(dim=0, index=label)
        distance = feature.dist(nCenter)
        centerloss = (1 / 2.0 / batch_size) * distance
        out = feature.mm(self.weight)
        ceLoss = F.cross_entropy(out, label)
        return out, ceLoss + self.lamda * centerloss
# 假设有 10 个类别，特征维度是 2
num_classes = 10
feat_dim = 2

# 创建一个简单的模型
class EModel(nn.Module):
    def __init__(self):
        super(EModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),  # 28*28*1 -> 28*28*32
            nn.BatchNorm2d(32),  # 原始不存在
            nn.ReLU(),
            # nn.Conv2d(3,6,5), #32*32*3->28*28*6
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),  # 28*28*32 -> 28*28*32
            nn.BatchNorm2d(32),  # 原始不存在
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 14*14*32
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),  # 14*14*64
            nn.BatchNorm2d(64),  # 原始不存在
            nn.ReLU(),
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),  # 14*14*64
            nn.BatchNorm2d(64),  # 原始不存在
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 7*7*16
        )
        self.layer5 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),  # 7*7*64
            nn.BatchNorm2d(128),  # 原始不存在
            nn.ReLU(),
        )
        self.layer6 = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),  # 7*7*128
            nn.BatchNorm2d(128),  # 原始不存在
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  # 3*3*128
        )
        self.layer7 = nn.Sequential(
            nn.Linear(3 * 3 * 128, 120),
            nn.Linear(120, 84),
            nn.Linear(84, 2, bias=False),
            # nn.LeakyReLU(),
            # nn.ReLU
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)  # 展开成batch-size个1维数组
        feature = self.layer7(x)
        return feature


# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root=r'F:\dataset\mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建模型、损失函数和优化器
model = EModel()
head = CenterLoss()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001}, {'params': head.parameters(), 'lr': 0.001}])

# 训练模型
num_epochs = 10
def train():
    model.to(device)
    head.to(device)
    model.train()
    head.train()
    precision = '0'
    for epoch in range(num_epochs):
        pre_tn = 0
        sam_nu = 0
        loss_center, loss_softmax = 0,0 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 获取特征向量
            features = model(images)
            out,loss = head(features,labels)
           
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            pre_label = torch.argmax(out, 1)
            pre_tn += torch.sum(pre_label==labels)
            sam_nu += len(labels)
        precision = f'{pre_tn/sam_nu:.4f}'
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, pre: {precision}')
        # print(f'{center_loss.centers}')
        state = {
            'state_dict': model.state_dict(),
            'head': head.state_dict(),
            'precision': precision
        }
    torch.save(state, f'checkpoint/center_loss_mnist_{precision}.pt')


def visualize():
    state = torch.load('checkpoint/center_loss_mnist_0.9949.pt')
    model = EModel()
    model.load_state_dict(state['state_dict'])
    # 提取中间层特征嵌入
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in train_loader:
            x = model(images)
            features.append(x)
            labels.append(label)

    features_2d = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # state['head']


    # 可视化
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=1)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
    plt.xlabel('Activation of the 1st neuron')
    plt.ylabel('Activation of the 2nd neuron')
    plt.title('Visualization of MNIST Features')
    plt.show()


# train()

visualize()
