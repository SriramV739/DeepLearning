import torch

from pathlib import Path
import numpy as np
from torchvision import datasets
from urllib.request import urlretrieve
from torchvision.transforms import v2
from torch.utils.data import DataLoader
import torch.nn as nn

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers=nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28,512),
            nn.ReLU(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self,x):
        return self.layers(x)



model=MLP()

loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=0.01)


#training loop

EPOCH=10
for epoch in range(EPOCH):
    for batch, (X,Y) in enumerate(train_dataloader):
        pred=model(X)
        loss1=loss(pred,Y)
        
        loss1.backward()
        optimizer.step()
        optimizer.zero_grad()

        if(batch%100==0):
            print(f"batch: {batch}, loss:{loss1}")


#test

model.eval()
correct=0
total=len(test_dataloader.dataset)
with torch.no_grad():
    for X,y in test_dataloader:
        pred=model(X)
        correct+=(pred.argmax(1)==y).type(torch.float).sum().item()


print(f"Accuracy: {correct/total}")