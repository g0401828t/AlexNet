import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torch.nn as nn
import torch.functional as F
from torch import optim

import os
import numpy as np
from datetime import datetime
from torchsummary import summary

from model import AlexNet
from plotgraph import plotgraph




### parameters
model_name = "AlexNet"
lr = 1e-4
momentum = 0 # 0.9
weight_dacay = 0 # 0.0005
batch_size = 256
epochs = 100
path = "D:/projects"
datapath = path + '/dataset'
resultpath = path + "/" + model_name + "/results"
modelpath = path + "/" + model_name + "/models/" + model_name + "_best_model.h"
if not os.path.exists(resultpath):
    os.mkdir(resultpath)
if not os.path.exists(modelpath):
    os.mkdir(modelpath)


### 사용 가능한 gpu 확인 및 설정
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(device)



### dataset 불러오기
# load STL10 train dataset, and check
data_transformer = transforms.Compose([transforms.ToTensor()])
train_set = datasets.STL10(datapath, split='train', download=True, transform=data_transformer)
print(train_set.data.shape)

# image Normalization
# meanRGB = [np.mean(x.numpy(), axis=(1,2)) for x, _ in train_set]
# stdRGB = [np.std(x.numpy(), axis=(1,2)) for x, _ in train_set]
# meanR = np.mean([m[0] for m in meanRGB])
# meanG = np.mean([m[1] for m in meanRGB])
# meanB = np.mean([m[2] for m in meanRGB])
# stdR = np.mean([s[0] for s in stdRGB])
# stdG = np.mean([s[1] for s in stdRGB])
# stdB = np.mean([s[2] for s in stdRGB])
# print("mean RGB:", meanR, meanG, meanB)
# print("std RGB:", stdR, stdG, stdB)
meanR, meanG, meanB = 0.4467106, 0.43980986, 0.40664646
stdR, stdG, stdB = 0.22414584, 0.22148906,  0.22389975
# define the image transformation for train_set
# in paper, using FiveCrop, normalize, horizontal reflection
train_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(227),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
])
# apply transformation to train_set and test_set
train_set.transform = train_transformer



# there is no validation set in STL10 dataset,
# make validation set by splitting the train set.
from sklearn.model_selection import StratifiedShuffleSplit
# StratifiedShuffleSplit splits indices of train in same proportion of labels
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
indices = list(range(len(train_set)))
y_train0 = [y for _,y in train_set]
for train_index, val_index in sss.split(indices, y_train0):
    print('train :', len(train_index) , 'val :', len(val_index))
# create two datasets from train_set
from torch.utils.data import Subset
val_set = Subset(train_set, val_index)
train_set = Subset(train_set, train_index)
print(len(train_set), len(val_set))
# count the number of images per calss in train_set and val_set
# import collections
# y_train = [y for _, y in train_set]
# y_val = [y for _, y in val_set]
# counter_train = collections.Counter(y_train)
# counter_val = collections.Counter(y_val)
# print(counter_train)
# print(counter_val)


### dataloader for train_set, val_set
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
# check dataloader
# for x,y in train_loader:
#     print(x.shape)
#     print(y.shape)
#     break
# for a,b in val_loader:
#     print(a.shape)
#     print(b.shape)
#     break


### bring model from model.py
model = AlexNet()
# model to cuda if available
model.to(device)
# print("model set to",next(model.parameters()).device)
# check model summary
# summary(model, input_size=(3, 227, 227))


### define Loss Function, Optimizer, Scheduler
criterion = nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(model.parameters(), lr = lr)
# optimizer = optim.SGD(model.parameters(), lr = lr, momentum=momentum, weight_decay=weight_dacay)
lr_sche = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

### training function with validation per epoch
def train():
    loss_list, valloss_list, valacc_list = [], [], []  # for ploting graph
    es = 5  # for early stopping
    for epoch in range(epochs):
        print("learning_rate:", lr)
        model.train()
        avg_loss, val_loss = 0, 0
        best_acc = 0
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()

            # forward propagation
            hypothesis = model(x)
            loss = criterion(hypothesis, y)

            # back propagation
            loss.backward()
            optimizer.step()

            avg_loss += loss
        avg_loss = avg_loss/len(train_loader)  # calculate average loss per epoch

        ### validation
        model.eval()
        with torch.no_grad():
            for i, (x, y) in enumerate(val_loader):
                x, y = x.to(device), y.to(device)

                prediction = model(x)
                # calculate validation Loss
                val_loss += criterion(prediction, y) / len(val_loader)
                # calculate validation Accuracy
                correct = prediction.max(1)[1] == y
                val_acc = correct.float().mean()

                # Early Stop
                if val_acc > best_acc:
                    best_acc = val_acc
                    es = 5
                    torch.save(model.state_dict(), modelpath)
                else: 
                    es -= 1
                if es == 0: 
                    model.load_state_dict(torch.load(modelpath))
                    break
            
            # for plotting graph
            loss_list.append(avg_loss.item())
            valloss_list.append(val_loss.item())
            valacc_list.append(val_acc.item())
        print(datetime.now().time().replace(microsecond=0), "EPOCHS: [{}/{}], avg_loss: [{:.4f}], val_acc: [{:.2f}%]".format(
                epoch+1, epochs, avg_loss.item(), val_acc.item()*100))
        plotgraph(loss_list=loss_list, valloss_list=valloss_list, valacc_list=valacc_list, path = resultpath)

# train
print("start training")
train()
