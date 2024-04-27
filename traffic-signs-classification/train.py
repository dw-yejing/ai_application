import os

# os.chdir("/content/drive/MyDrive/traffic-signs-classification/")

import shutil

from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

from model.model import *
from torch import nn
import torch.utils.data as Data

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cwd = os.getcwd()
learning_rate = 1e-2
train_step = 0
# 添加tensorboard
writer = SummaryWriter("logs_train")

train_data = None
train_data_loader = None
validation_data = None
validation_data_loader = None


def data_split(splited: bool):
    if not splited:
        return
    data_dir = os.path.join(cwd, "data", "train")
    all_data = ImageFolder(data_dir)

    train_size = int(len(all_data) * 0.8)
    validation_size = len(all_data) - train_size
    train_set, validation_set = torch.utils.data.random_split(all_data, [train_size, validation_size])

    # copy file
    splited_data_dir = os.path.join(cwd, "data", "splited")
    splited_train_data_dir = os.path.join(splited_data_dir, "train")
    splited_validation_data_dir = os.path.join(splited_data_dir, "validation")
    if not os.path.isdir(splited_data_dir):
        os.mkdir(splited_data_dir)
        os.mkdir(splited_train_data_dir)
        os.mkdir(splited_validation_data_dir)
    else:
        return
    samples = train_set.dataset.samples
    classes = train_set.dataset.classes
    for i in range(len(train_set)):
        img, label = samples[train_set.indices[i]]
        label_dir = os.path.join(splited_train_data_dir, classes[label])
        if not os.path.isdir(label_dir):
            os.mkdir(label_dir)
        shutil.copy(img, label_dir)

    for i in range(len(validation_set)):
        img, label = samples[validation_set.indices[i]]
        label_dir = os.path.join(splited_validation_data_dir, classes[label])
        if not os.path.isdir(label_dir):
            os.mkdir(label_dir)
        shutil.copy(img, label_dir)

    print("============ 数据分割完成 ============")


def data_preprocess():
    # data preprocessing
    global train_data
    global train_data_loader
    global validation_data
    global validation_data_loader
    train_data_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 224
        transforms.RandomHorizontalFlip(),  # 默认概率0.5
        transforms.ToTensor(),
        # 图像标准化处理
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_data_dir = os.path.join(cwd, "data", "splited", "train")
    validation_data_dir = os.path.join(cwd, "data", "splited", "validation")
    train_data = ImageFolder(train_data_dir, transform=train_data_transforms)
    validation_data = ImageFolder(validation_data_dir, transform=train_data_transforms)

    train_data_loader = Data.DataLoader(train_data, batch_size=64, shuffle=True, num_workers=2)
    validation_data_loader = Data.DataLoader(validation_data, batch_size=64, shuffle=True, num_workers=2)


net = None
criterion = None
optimizer = None
scheduler = None
best_accuracy = 0
start_epoch = 0


def build_model(resume):
    # building model
    global net
    global criterion
    global optimizer
    global scheduler
    global best_accuracy
    global start_epoch

    print('============ Building model ============')
    net = Dawei()
    net = net.to(device)
    # if device == 'cuda':
    #     net = torch.nn.DataParallel(net)

    if resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(os.path.join(cwd, 'checkpoint', 'ckpt.pth'))
        net.load_state_dict(checkpoint['net'])
        best_accuracy = checkpoint['accuracy']
        start_epoch = checkpoint['epoch'] + 1

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, momentum=0.5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


def train(epoch):
    global train_step
    train_accuracy = 0
    train_loss = 0
    net.train()
    epoch_step = 0
    for index, (imgs, labels) in enumerate(train_data_loader):
        imgs, labels = imgs.to(device), labels.to(device)

        outputs = net(imgs)
        loss = criterion(outputs, labels)
        pre_labels = torch.argmax(outputs, 1)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_step = train_step + 1
        epoch_step += 1
        train_loss += loss.item()
        train_accuracy += torch.sum(pre_labels == labels)
        if train_step % 5 == 0:
            print("训练次数：{}, Loss: {}, ".format(train_step, train_loss / (index + 1)))
            print("训练次数：{}, Accuracy: {}, ".format(train_step, train_accuracy / epoch_step))
            writer.add_scalar("train_loss", loss.item(), train_step)
            writer.add_scalar("train_accuracy", train_accuracy / train_step, train_step)
        # writer.add_scalar("train_accuracy", train_accuracy / len(train_data), epoch)


def validate(epoch):
    # validation
    global best_accuracy
    net.eval()
    validation_loss = 0
    validation_accuracy = 0
    with torch.no_grad():
        for imgs, labels in validation_data_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = net(imgs)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            pre_labels = torch.argmax(outputs, 1)
            validation_accuracy += torch.sum(pre_labels == labels)

    print("第{}轮,验证集的Loss: {}".format(epoch, validation_loss))
    print("第{}轮,验证集的正确率: {}".format(epoch, validation_accuracy / len(validation_data)))
    writer.add_scalar("validation_loss", validation_loss, epoch)
    writer.add_scalar("validation_accuracy", validation_accuracy / len(validation_data), epoch)

    # Save checkpoint.
    accuracy = 100. * validation_accuracy / len(validation_data)
    if accuracy > best_accuracy:
        print('Saving model')
        state = {
            'net': net.state_dict(),
            'accuracy': accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, os.path.join(cwd, 'checkpoint', 'ckpt.pth'))
        best_accuracy = accuracy
        print("Saved model")


if __name__ == '__main__':
    data_preprocess()
    build_model(False)
    for i in range(start_epoch, start_epoch + 120):
        print("-------第 {} 轮训练开始-------".format(i + 1))
        train(i)
        validate(i)
        scheduler.step()
        print("学习率:{}".format(optimizer.state_dict()['param_groups'][0]['lr']))
    writer.close()

