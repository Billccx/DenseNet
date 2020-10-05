import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Transition(nn.Module):
    def __init__(self,In,Out):
        super(Transition, self).__init__()
        self.bn=nn.BatchNorm2d(num_features=In)
        self.conv=nn.Conv2d(in_channels=In,out_channels=Out,kernel_size=1,bias=False)
    def forward(self,x):
        #y=self.bn(x)
        #y=F.relu(y,inplace=True)
        y=self.conv(F.relu(self.bn(x)))
        #y=F.avg_pool2d(y,kernel_size=2,stride=2)
        y = F.avg_pool2d(y,2)
        return y

class Bottleneck(nn.Module):
    def __init__(self,In,growthrate):
        super(Bottleneck, self).__init__()
        self.bn1=nn.BatchNorm2d(num_features=In)
        self.conv1=nn.Conv2d(in_channels=In,out_channels=4*growthrate,kernel_size=1,bias=False)
        self.bn2=nn.BatchNorm2d(num_features=4*growthrate)
        self.conv2=nn.Conv2d(in_channels=4*growthrate,out_channels=growthrate,kernel_size=3,padding=1,bias=False)

    def forward(self,x):
        '''
        y=self.bn1(x)
        y=F.relu(y,inplace=True)
        y=self.conv1(y)
        y=self.bn2(y)
        y = F.relu(y,inplace=True)
        y=self.conv2(y)
        ret=torch.cat((x,y),dim=1)
        return ret
        '''
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat((x, out), 1)
        return out

class DenseNet(nn.Module):
    def __init__(self,numClasses,growthrate,numlayers,compression):
        super(DenseNet, self).__init__()
        nChannels=growthrate*2

        self.conv1=nn.Conv2d(in_channels=3,out_channels=nChannels,kernel_size=3,padding=1,bias=False)

        self.DenseBlock1=self.BuildDense(nChannels,growthrate,numlayers)
        nChannels+=numlayers*growthrate
        OutChannels=int(math.floor(compression*growthrate))
        self.Trans1=Transition(nChannels,OutChannels)
        nChannels=OutChannels

        self.DenseBlock2=self.BuildDense(nChannels,growthrate,numlayers)
        nChannels += numlayers * growthrate
        OutChannels = int(math.floor(compression * growthrate))
        self.Trans2 = Transition(nChannels, OutChannels)
        nChannels = OutChannels

        self.DenseBlock3 = self.BuildDense(nChannels, growthrate, numlayers)
        nChannels += numlayers * growthrate

        self.bn=nn.BatchNorm2d(num_features=nChannels)
        self.fc=nn.Linear(in_features=nChannels,out_features=numClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def BuildDense(self,In,growthrate,numlayers):
        layers=[]
        InChannels=In
        for i in range(numlayers):
            layers.append(Bottleneck(InChannels,growthrate))
            InChannels+=growthrate
        return nn.Sequential(*layers)

    def forward(self,image):
        '''
        y = self.conv1(image)
        #y = self.DenseBlock1(y)
        y = self.Trans1(self.DenseBlock1(y))
        #y = self.DenseBlock2(y)
        y = self.Trans2(self.DenseBlock2(y))
        y = self.DenseBlock3(y)

        y = self.bn(y)
        y = F.relu(y, inplace=True)
        y = F.avg_pool2d(y, 8)
        y = y.view(y.shape[0],-1)
        y = self.fc(y)
        ret = F.log_softmax(y)
        '''
        out = self.conv1(image)
        out = self.Trans1(self.DenseBlock1(out))
        out = self.Trans2(self.DenseBlock2(out))
        out = self.DenseBlock3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn(out)), 8))
        ret = F.log_softmax(self.fc(out))
        return ret

#net=DenseNet(10,32,2,1)


