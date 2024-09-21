import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device

        # 初始化类中心
        self.centers = nn.Parameter(10*torch.Tensor(10, 2))

    def forward(self, x, labels):
        # 获取每个样本对应的类中心
        centers_batch = self.centers.index_select(0, labels)

        # 计算类内距离
        loss = F.mse_loss(x, centers_batch)
        return loss

# 假设有 10 个类别，特征维度是 2
num_classes = 10
feat_dim = 2

# 创建一个简单的模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.feat = nn.Sequential(
            nn.Linear(28 * 28, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            nn.Linear(64, 2)
        )
        self.head = nn.Parameter(torch.Tensor(2, 10))
        nn.init.kaiming_uniform_(self.head)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.feat(x)
        x = x.mm(self.head)
        return x

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root=r'F:\dataset\mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 创建模型、损失函数和优化器
model = SimpleModel()
center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam([{'params': model.parameters(), 'lr': 0.001}, {'params': center_loss.parameters(), 'lr': 0.001}])

# 训练模型
num_epochs = 40
def train():
    model.to(device)
    model.train()
    center_loss.to(device)
    center_loss.train()
    precision = '0'
    for epoch in range(num_epochs):
        pre_tn = 0
        sam_nu = 0
        loss_center, loss_softmax = 0,0 
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 获取特征向量
            features = model(images)
            x = images.view(-1, 28 * 28)
            feat_2ds = model.feat(x)

            # 计算损失
            loss_center = center_loss(feat_2ds, labels)
            loss_softmax = criterion(features, labels)
            loss = 0.1*loss_softmax +  loss_center
            # loss =  loss_center

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 计算准确率
            pre_label = torch.argmax(features, 1)
            pre_tn += torch.sum(pre_label==labels)
            sam_nu += len(labels)
        precision = f'{pre_tn/sam_nu:.4f}'
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, pre: {precision}, loss_center: {loss_center.item()}, loss_softmax: {loss_softmax.item()}')
        # print(f'{center_loss.centers}')
        state = {
            'state_dict': model.state_dict(),
            'head': center_loss.state_dict(),
            'precision': precision
        }
    torch.save(state, f'checkpoint/center_loss_mnist_{precision}.pt')


def visualize():
    state = torch.load('checkpoint/center_loss_mnist_0.9306.pt')
    model = SimpleModel()
    model.load_state_dict(state['state_dict'])
    # 提取中间层特征嵌入
    model.eval()
    features = []
    labels = []
    center_features = []
    with torch.no_grad():
        for images, label in train_loader:
            x = images.view(-1, 28 * 28)
            x = model.feat(x)
            features.append(x)
            labels.append(label)
        
        centers = state['head']['centers'].cpu()
        center_features.append(centers)


    features_2d = torch.cat(features).numpy()
    features_center = torch.cat(center_features).numpy()
    labels = torch.cat(labels).numpy()

    # state['head']


    # 可视化
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab20', s=1)
    plt.scatter(features_center[:, 0], features_center[:, 1], c='white', s=5)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
    plt.xlabel('Activation of the 1st neuron')
    plt.ylabel('Activation of the 2nd neuron')
    plt.title('Visualization of MNIST Features')
    plt.show()


# train()

visualize()
