import torch
from torch import nn
from torchvision import models
from torchinfo import summary

print(torch.__version__)

class TwoStreamModel(nn.Module):
    def __init__(self, num_classes):
        super(TwoStreamModel, self).__init__()
        
        #spatial stream
        self.spatial_net=models.resnet101(pretrained=True)
        self.spatial_net.fc=nn.Identity()

        #temporal stream
        self.temporal_net=models.resnet101(pretrained=True)
        old_conv1=self.temporal_net.conv1
        new_conv1=nn.Conv2d(20,64,kernel_size=7,stride=2, padding=3,bias=False)
        with torch.no_grad():
            avg_weight=old_conv1.weight.mean(dim=1,keepdim=True)
            new_conv1.weight.copy_(avg_weight.repeat(1,20,1,1))
        self.temporal_net.conv1=new_conv1
        self.temporal_net.fc=nn.Identity()

        #calssifier
        self.classifier=nn.Linear(2048*2,num_classes)

        #doprout
        self.dropout=nn.Dropout(p=0.8)

    def  forward(self, spatial_input,temporal_input):
        
        #conv and flatten spatial features
        spatial_features=self.spatial_net(spatial_input)
        spatial_features=torch.flatten(spatial_features,1)

        #conv and flatten temporal fratures
        temporal_features=self.temporal_net(temporal_input)
        temporal_features=torch.flatten(temporal_features,1)

        #cat
        featuers=torch.cat((spatial_features,temporal_features),dim=1)
        #dropout
        featuers=self.dropout(featuers)
        #classifier
        predict=self.classifier(featuers)

        return predict

if __name__ == '__main__':
    batch_size=16
    spatial_input_size=(batch_size,3,224,224)
    temporal_input_size=(batch_size,20,224,224)
    model=TwoStreamModel(num_classes=2)
    summary(model,input_data=[torch.randn(spatial_input_size),torch.randn(temporal_input_size)])