import torch
import torch.nn as nn
from torchsummary import summary
from torchvision.models import resnet101



class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.txt_network = nn.Sequential(nn.Linear(768,512),
                                         nn.BatchNorm1d(512),
                                         nn.ReLU(),
                                         nn.Linear(512,256),
                                         nn.BatchNorm1d(256),
                                         nn.ReLU(),
                                         nn.Linear(256,128),
                                         nn.Softmax(dim=-1))
        # self.img_network = resnet101(weights='ResNet101_Weights.DEFAULT')
        # self.merge_network = nn.Sequential(nn.Linear(1000+64,512),
        #                                    nn.ReLU(),
        #                                    nn.Linear(512,256),
        #                                    nn.ReLU(),
        #                                    nn.Linear(256,128),
        #                                    nn.ReLU(),
        #                                    nn.Softmax(dim=-1))

    def forward(self,txt,img):
        txt = self.txt_network(txt)
        # img = self.img_network(img)
        # x = torch.cat([txt,img], dim=-1)
        # y = self.merge_network(x)
        return txt #y
    

if __name__ == '__main__':
    model = MyModel()
    txt = torch.zeros(16,768)
    img = torch.zeros(16,3,256,256)
    y_pred = model(txt,img)
    print(y_pred.shape)

    summary(model, [(768,), (3,256,256)], device = 'cpu')