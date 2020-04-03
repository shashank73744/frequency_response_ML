import math
import random
import numpy as np
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import pandas as pd    
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from models import MLP,NALU,NAC,LIN

import torch.utils.data as data_utils


inputData = np.load('/Users/shashanksaurabh/Desktop/MS/On-campus/NALU-pytorch-master/ExerciseData/InputData.npy')
outputData = np.load('/Users/shashanksaurabh/Desktop/MS/On-campus/NALU-pytorch-master/ExerciseData/OutputData.npy')
X_train, y_train, X_test, y_test = torch.Tensor(inputData[:60000]), torch.Tensor(outputData[:60000]), torch.Tensor(inputData[60000:]), torch.Tensor(outputData[60000:])


NORMALIZE = True
HIDDEN_DIM = 800
LEARNING_RATE = .003
NUM_ITERS = int(1e2)
RANGE = [5, 10]

class DataSet(Dataset):

    def __init__(self):
        self.inputData = np.load('/Users/shashanksaurabh/Desktop/MS/On-campus/NALU-pytorch-master/ExerciseData/InputData.npy')
        self.outputData = np.load('/Users/shashanksaurabh/Desktop/MS/On-campus/NALU-pytorch-master/ExerciseData/OutputData.npy')
        self.inputData = torch.Tensor(self.inputData[:60000])
        self.outputData = torch.Tensor(self.outputData[:60000])

    def __len__(self):
        return len(self.inputData)

    def __getitem__(self, idx):
        inputVector = self.inputData[idx]
        outputVector = self.outputData[idx]
        return inputVector,outputVector

def load_checkpoint(model, optimizer, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, start_epoch


def train(model, optimizer, data, target,_batch_size):
    obj = DataSet()
    train_loader = torch.utils.data.DataLoader(obj,batch_size=_batch_size)
    start_epoch = 0
    #Comment these line while running model other than MLP
    model,optimizer,start_epoch = load_checkpoint(model, optimizer,"/Users/shashanksaurabh/Desktop/MS/On-campus/NALU-pytorch-master/model7_lr_3.py")
    epoch = start_epoch
    while True:
        epoch = epoch +1
        _loss = 0
        _mea = 0
        for idx ,batch in enumerate(train_loader):
            inputVector,outputVector = batch
            out = model(inputVector)
            loss = F.mse_loss(out, outputVector)
            mea = torch.mean(torch.abs(outputVector - out))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #Comment these line while running model other than MLP
        if epoch%5 == 0:
            state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),
             'optimizer': optimizer.state_dict() }
            torch.save(state, "/Users/shashanksaurabh/Desktop/MS/On-campus/NALU-pytorch-master/model7_lr_3.py")
        print("-------------\t{}: loss: {:.7f} - mea: {:.7f}".format(
                epoch, loss.item(), mea.item())
            )


def test(model, data, target):
    with torch.no_grad():
        out = model(data)
        return torch.abs(target - out)



def main():
    __model = MLP(
            in_dim=400,
            hidden_dim=HIDDEN_DIM,
            out_dim=800
        )

    print("\tTraining {}...".format(__model.__str__().split("(")[0]))
    optim = torch.optim.Adam(__model.parameters(), lr=LEARNING_RATE)
    train(__model, optim, X_train, y_train,128)
    mse = test(__model, X_test, y_test).mean().item()
    print(mse)


if __name__ == '__main__':
    main()
