import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import datetime
import os
import argparse
import time
from models import *
from utils import progress_bar
import zipfile
import wget
from torchinfo import summary
from matplotlib import pyplot as plt

from ptflops import get_model_complexity_info


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def plot_kernels(tensor, num_cols=6):
    if not tensor.ndim==4:
        raise Exception("assumes a 4D tensor")
    if not tensor.shape[-1]==3:
        raise Exception("last dim needs to be 3 to plot")
    num_kernels = tensor.shape[0]
    num_rows = 1+ num_kernels // num_cols
    fig = plt.figure(figsize=(num_cols,num_rows))
    for i in range(tensor.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        ax1.imshow(tensor[i])
        ax1.axis('off')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

    plt.subplots_adjust(wspace=0.1, hspace=0.1)

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--ds', default='cifar10', help='Dataset to train on')
parser.add_argument('--wd', default=5e-4, type=float, help='weight decay')
parser.add_argument('--sn', default='VGG16_rand', help='File Save Name')
parser.add_argument('--epochs', default=200, type=int, help='Total number of epochs')
parser.add_argument('--bs', default=128, type=int, help='Batch Size')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'



best_acc = 0  # best test accuracy
best_e = -1  # epoch where best accuracy was achieved


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main_loop():
    global best_acc
    best_acc=0.
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    seed_everything()
    # Data
    print('==> Preparing data..')

    if(dataset == 'cifar10' or dataset == 'cifar100'):
        imW = 32
        imH = 32
        skip = 4
    else:
        imW = 224
        imH = 224
        skip = 4

    if(dataset == 'tinyimagenet'):
        imW = 64
        imH = 64
        skip = 4

    transform_train = transforms.Compose([
        transforms.Resize((imW,imW)),
        transforms.RandomCrop(imW,padding=skip),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize((imW,imW)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])


    if(dataset == 'cifar10'):
        num_classes = 10
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=2)

        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    if (dataset == 'cifar100'):
        num_classes=100
        trainset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=2)
        classes = ('plane', 'car', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck')

    if (dataset == 'tinyimagenet'):
        if(os.path.isfile('./data/tiny-imagenet-200.zip') == False):
            wget.download('http://cs231n.stanford.edu/tiny-imagenet-200.zip','./data/')
        if(os.path.isfile('./data/imagenet16') == False):
            with zipfile.ZipFile('./data/tiny-imagenet-200.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/')
        num_classes=200
        trainset = torchvision.datasets.ImageFolder(root='./data/TinyImageNet/train/',transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='./data/TinyImageNet/val/', transform=transform_test)
        trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=4, shuffle=True)
        testloader = DataLoader(testset, batch_size=args.bs, num_workers=4, shuffle=False)

    if (dataset == 'imagenet16'):
        if(os.path.isfile('./data/imagenet16.zip') == False):
            wget.download('https://zenodo.org/records/8027520/files/imagenet16.zip','./data/')
        if(os.path.isfile('./data/imagenet16') == False):
            with zipfile.ZipFile('./data/imagenet16.zip', 'r') as zip_ref:
                zip_ref.extractall('./data/')
        num_classes=16
        trainset = torchvision.datasets.ImageFolder(root='./data/imagenet16/train',transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root='./data/imagenet16/val', transform=transform_test)
        trainloader = DataLoader(trainset, batch_size=args.bs, num_workers=2, shuffle=True)
        testloader = DataLoader(testset, batch_size=args.bs, num_workers=2, shuffle=False)

    print('==> Building model..')
    if(args.sn=='VGG16'):
        net = VGG('VGG16',num_classes = num_classes)
    if(args.sn=='VGG16_step'):
        net = VGG_step(vgg_name='VGG16',num_classes = num_classes)
    if (args.sn == 'mobilenetv2'):
        net = MobileNetV2(num_classes = num_classes)
    if (args.sn == 'mobilenetv2_step'):
        net = MobileNetV2_Step(num_classes = num_classes)
    if(args.sn=='resnet50'):
        net = ResNet50(num_classes = num_classes)
    if(args.sn=='resnet50_step'):
        net = ResNet50_step(num_classes = num_classes)
    if (args.sn == 'efficientnetb0'):
        net = EfficientNetB0(num_classes = num_classes)
    if (args.sn == 'efficientnetb0_step'):
        net = EfficientNetB0_step(num_classes = num_classes)
    if (args.sn == 'stepnet'):
        net = StepNet(num_classes = num_classes)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=0.9, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=args.epochs)
    summary_str=summary(net, (1,3, imW, imH),col_names =("input_size", "output_size", "num_params", "kernel_size", "mult_adds"),depth=8,verbose =0)
    
    macs,params = get_model_complexity_info(net,(3, imW, imH),as_strings=True, print_per_layer_stat=False, verbose=False)
    print(args.sn)
    print('{:<30}  {:<8}'.format('Computational complexity:',macs))
    print('{:<30}  {:<8}'.format('Number of params:', params))
    print(get_n_params(net))
    summary_str.non_trainable_params = summary_str.total_params - summary_str.trainable_params
    print('Total Params:',summary_str.total_params)
    print('Trainable Params:',summary_str.trainable_params)
    print('Non-Trainable Params:', summary_str.non_trainable_params)
    summary_str.param_bytes = (summary_str.trainable_params * 32 + summary_str.non_trainable_params * 2) / 8 / 1000000
    print('Bytes for Params:',summary_str.param_bytes)

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Write the summary to a text file
    with open('./saved_models/'+dataset+'_'+args.sn+'.txt', 'w',encoding='utf-8') as f:
        f.write(str(summary_str))

    # Start the timer
    start_time = time.time()
    for epoch in range(start_epoch, start_epoch + args.epochs):
        train(epoch,net,optimizer,criterion,trainloader)
        test(epoch,net,criterion,testloader)
        scheduler.step()
    end_time = time.time()
    # Calculate the elapsed time
    execution_time = end_time - start_time

    print(f"Execution time: {execution_time} seconds")

    with open("./metrics/metrics_" + dataset + ".txt", "a") as file:
        # Append new text to the file
        file.write(f'{args.sn:^50}: {best_acc:^4.6f}   {summary_str.total_params:^20}   {summary_str.trainable_params:^20}   {summary_str.non_trainable_params:^20}   {summary_str.param_bytes:^20}\n')



# Training
def train(epoch,net,optimizer,criterion,trainloader):
    print('\nEpoch: %d/%d' % (epoch,args.epochs))
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for param_group in optimizer.param_groups:
        print("Learning Rate:",param_group['lr'])

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

def test(epoch,net,criterion,testloader,save=True):
    global best_acc,best_e,fname,dataset
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | Best epoch:%d(%.3f%%)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total,best_e,best_acc))

    # Save checkpoint.
    if(save):
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
			   'net':net,
               'acc': acc,
               'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './saved_models/'+dataset+'_'+fname+'.pth')
            best_acc = acc
            best_e = epoch

if __name__ == '__main__':

    now = datetime.datetime.now()
    datastr = now.strftime("%Y-%m-%d %H:%M:%S")
    fname = args.sn
    dataset = args.ds
    from_list = True


    with open("./metrics/metrics_" + dataset + ".txt", "a") as file:
       # Append new text to the file
       file.write(f'\n\n{dataset:^50}  {datastr} | batch: {args.bs}, Learning Rate: {args.lr}, Weight Decay: {args.wd}, Total epochs: {args.epochs}  \n\n')
       file.write(f'{"Model Name":^50}: {"Accuracy":^10}   {"Total Parameters":^20}   {"Trainable Parameters":^20}   {"Non-Trainable Parameters":^20}   {"MBytes for Params":^20}\n')


    main_loop()

