import torch.nn as nn

class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=2)
         
        self.conv2 = nn.Conv2d(in_channels=32, 
                                      out_channels=64, 
                                      kernel_size=(3,3), 
                                      stride=2)       
        
        self.conv3 = nn.Conv2d(in_channels=64, 
                                      out_channels=128, 
                                      kernel_size=(3,3),
                                      stride=2)
       
        self.gloabl_pool = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(in_features=128, out_features=8)
        
    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU(True)(x)
        x = self.conv2(x)
        x = nn.ReLU(True)(x)
        x = self.conv3(x)
        x = nn.ReLU(True)(x)
        x = self.gloabl_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x
