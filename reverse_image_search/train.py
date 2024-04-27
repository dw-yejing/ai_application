import torch
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import matplotlib.pyplot as plt
from dataset import pairs_train_reader
from model import MyNet


def draw_loss(loss_arr):
    plt.title("training loss", fontsize=24)
    plt.xlabel("batch", fontsize=14)
    plt.ylabel("loss", fontsize=14)
    plt.plot([i for i in range(len(loss_arr))], loss_arr, color='red', label='training loss')
    plt.legend()
    plt.grid()
    plt.show() 

def train(model):
    print('start training ... ')
    model.train()

    inverse_temperature = torch.from_numpy(np.array([1.0/0.2], dtype='float32')).to(device)

    epoch_num = 50
    
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    model.train()

    loss_arr=[]
    best_model = {
        "epoch": 0,
        "lr": 0.001,
        "loss": 1000,
        "model": model.state_dict()
    }
    for epoch in range(epoch_num):
        for batch_id, data in enumerate(pairs_train_reader()):
            anchors, positives = data[0].to(device), data[1].to(device)

            anchor_embeddings = model(anchors)
            positive_embeddings = model(positives)
            
            similarities = torch.matmul(anchor_embeddings, positive_embeddings.t()) 
            similarities = torch.multiply(similarities, inverse_temperature)
            
            sparse_labels = torch.arange(0, num_classes, dtype=torch.int64).to(device)

            loss = loss_func(similarities, sparse_labels)

            loss_arr.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            if loss<best_model['loss']:
                # 暂存模型
                best_model['epoch']=epoch
                best_model['lr']=current_lr
                best_model['loss']=loss
                best_model['model']= model.state_dict()
                # 保存模型
                torch.save(best_model, 'model/best_model.pth')

            if batch_id % 500 == 0:
                print("epoch: {}, batch_id: {}, lr: {}, loss is: {}".format(epoch, batch_id, current_lr, loss.item()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()
     
    draw_loss(loss_arr)

num_classes = 10  
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = MyNet().to(device)
train(model)
