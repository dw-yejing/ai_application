import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


class L2NormalizedLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(L2NormalizedLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.s = 10

    def forward(self, x):
        normalized_weight = self.s * F.normalize(self.weight, p=2, dim=1)
        return F.linear(x, normalized_weight)

# 使用自定义层
model = L2NormalizedLinear(10, 5)


# 定义简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.feat = nn.Sequential(
            nn.Linear(28 * 28, 128),

            nn.Linear(128, 64),

            nn.Linear(64, 2)
        )
        self.head = nn.Parameter(torch.Tensor(2, 10))
        self.disable_bias(self.feat)

    def disable_bias(self, modules):
        for m in modules:
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias = None

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.feat(x)
        x = x.mm(self.head)
        return x

# 加载 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root=r'F:\dataset\mnist', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 实例化模型、定义损失函数和优化器
model = SimpleNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
num_epochs = 10
def train():
    precision = '0'
    for epoch in range(num_epochs):
        pre_tn = 0
        sam_nu = 0
        for images, labels in train_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 计算准确率
            pre_label = torch.argmax(outputs, 1)
            pre_tn += torch.sum(pre_label==labels)
            sam_nu += len(labels)
        precision = f'{pre_tn/sam_nu:.4f}'
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, pre: {precision}')
    state = {
        'state_dict': model.state_dict(),
        'precision': precision
    }
    torch.save(state, f'checkpoint/softmax_loss_mnist_{precision}.pt')

def visualize():
    state = torch.load('checkpoint/softmax_loss_mnist_0.8957.pt')
    model = SimpleNN()
    model.load_state_dict(state['state_dict'])
    # 提取中间层特征嵌入
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for images, label in train_loader:
            x = images.view(-1, 28 * 28)
            x = model.feat(x)
            features.append(x)
            labels.append(label)

    features_2d = torch.cat(features).numpy()
    labels = torch.cat(labels).numpy()

    # 使用 t-SNE 将高维特征嵌入降维至2维
    # from sklearn.manifold import TSNE
    # tsne = TSNE(n_components=2, random_state=42)
    # features_2d = tsne.fit_transform(features)

    # import umap
    # umap_reducer = umap.UMAP(n_components=2, random_state=42)
    # features_2d = umap_reducer.fit_transform(features)

    # from sklearn.decomposition import PCA

    # pca = PCA(n_components=2)
    # features_2d = pca.fit_transform(features)

    # 可视化
    plt.figure(figsize=(5, 5))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=1)
    plt.legend(handles=scatter.legend_elements()[0], labels=list(range(10)))
    plt.xlabel('Activation of the 1st neuron')
    plt.ylabel('Activation of the 2nd neuron')
    plt.title('Visualization of MNIST Features')
    plt.show()


train()

# visualize()