from reader import Reader
from model import CRNN
import torch
import os
import time

# 数据集路径设置
DATA_PATH = "./dataset/OCR_Dataset"
batch_size = 128 * 2
# 训练轮数
EPOCH = 50
device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 1e-3
train_data = Reader(DATA_PATH)
train_data_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, shuffle=True, num_workers=2
)


model = CRNN()
model.to(device)
loss_func = torch.nn.CTCLoss(blank=10)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimal_model = {"loss": 1000000, "stat": model.state_dict(), "lr": 0.1}


def train():
    global epoch_loss
    for index, (imgs, labels) in enumerate(train_data_loader):
        optimizer.zero_grad()
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        input_lengths = torch.full((labels.shape[0],), 8, dtype=torch.long)
        target_lengths = torch.full((labels.shape[0],), 4, dtype=torch.long)
        outputs = torch.transpose(outputs, 0, 1).log_softmax(2)

        loss = loss_func(outputs, labels, input_lengths, target_lengths)
        loss_value = loss.item()
        loss.backward()
        optimizer.step()
        # 打印训练信息
        epoch_loss += loss_value


def adjust_learning_rate():
    optimizer.state_dict()["param_groups"][0]["lr"] *= 0.1


adjust_epochs = []

if __name__ == "__main__":
    timestamp = int(time.time())
    for i in range(EPOCH):
        epoch_loss = 0
        print("-------第 {} 轮训练开始-------".format(i + 1))
        train()
        if i in adjust_epochs:
            adjust_learning_rate(optimizer)
        print(f'lr: {optimizer.state_dict()["param_groups"][0]["lr"]}')

        print(f"epoch: {i} loss: {epoch_loss}")
        with open(f"output/loss-{timestamp}", "a") as f:
            f.write(str(epoch_loss) + "\n")
        # 保存模型
        if epoch_loss < optimal_model["loss"] and i > 30:
            optimal_model["loss"] = epoch_loss
            optimal_model["stat"] = model.state_dict()
            optimal_model["lr"] = optimizer.state_dict()["param_groups"][0]["lr"]
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(
                optimal_model,
                os.path.join(os.getcwd(), "checkpoint", f"ckpt-{epoch_loss}.pth"),
            )
