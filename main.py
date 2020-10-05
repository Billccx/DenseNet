import torch
import torch.nn as nn
import densenet
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import argparse


def main():
    parser = argparse.ArgumentParser(description='DenseNet')
    parser.add_argument('--BatchSize', type=int, default=64)
    parser.add_argument('--Epochs', type=int, default=50)
    parser.add_argument('--save')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--opt', type=str, default='adam',choices=('sgd', 'adam'))
    args = parser.parse_args()
    args.cuda=True if torch.cuda.is_available() else False

    normMean = [0.49139968, 0.48215827, 0.44653124]
    normStd = [0.24703233, 0.24348505, 0.26158768]
    normTransform = transforms.Normalize(normMean, normStd)

    trainTransform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normTransform
    ])
    testTransform = transforms.Compose([
        transforms.ToTensor(),
        normTransform
    ])

    trainLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=True, download=False,transform=trainTransform),
        batch_size=args.BatchSize,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    testLoader = DataLoader(
        dset.CIFAR10(root='cifar', train=False, download=False,transform=testTransform),
        batch_size=args.BatchSize,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    net=densenet.DenseNet(numClasses=10,growthrate=32,numlayers=12,compression=0.5)
    if args.cuda:
        net = net.cuda()

    if args.opt == 'sgd':
        optimizer = optim.SGD(net.parameters(), lr=1e-1,momentum=0.9, weight_decay=1e-4)
    elif args.opt == 'adam':
        optimizer = optim.Adam(net.parameters(), weight_decay=1e-4)

    trainacc=[]
    testacc=[]
    for epoch in range(1,args.Epochs+1):
        train(args, epoch, net, trainLoader, optimizer, trainacc)
        validate(args, epoch, net, testLoader, testacc)

    xaxis=[i for i in range(1,args.Epochs+1)]
    plt.plot(xaxis, trainacc, color='red', label='train')
    plt.plot(xaxis, testacc, color='blue', label='test')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.title('Accruacy')
    plt.legend()
    plt.show()
    plt.savefig('result.jpg')


def train(args, epoch, net, trainLoader, optimizer, trainacc):
    net.train()
    hasDone=0
    lenEpoch=len(trainLoader.dataset)

    for index,(data,label) in enumerate(trainLoader):
        if args.cuda:
            data=data.cuda()
            label=label.cuda()

        optimizer.zero_grad()
        out=net(data)
        loss=F.nll_loss(out,label)
        loss.backward()
        optimizer.step()

        hasDone+=args.BatchSize
        pred = out.data.max(1)[1]

        correct=pred.eq(label).cpu().sum()
        acc=100.*correct/args.BatchSize
        progress=epoch-1+hasDone/lenEpoch
        print('Train Epoch: {:.2f} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAccruacy: {:.6f}'.format(
            progress, hasDone, lenEpoch, 100.*hasDone/lenEpoch,loss.data, acc))

        if(index==0):
            trainacc.append(acc)


def validate(args, epoch, net, testLoader, testacc):
    net.eval()
    testloss=0
    correct=0
    lendata = len(testLoader.dataset)
    numiter = len(testLoader)
    with torch.no_grad():
        for data,label in testLoader:

            if args.cuda:
                data=data.cuda()
                label=label.cuda()

            data.requires_grad=False
            label.requires_grad = False

            out=net(data)
            loss = F.nll_loss(out, label)

            pred = out.data.max(1)[1]
            correct += pred.eq(label).cpu().sum()
            #correct += pred.eq(label).sum()
            testloss+=loss
        torch.cuda.empty_cache()

    aveloss=testloss/numiter
    acc=int(correct.data)/lendata

    print('\nTest set: Average loss: {:.4f}, Accruacy: {}/{} ({:.6f}%)\n'.format(
        aveloss, correct, lendata, acc*100.))
    testacc.append(acc*100)


if __name__=='__main__':
    main()