import torch
from torch import nn
from torch import optim
from model import SRCNN
import torch.nn.functional as F
from dataloader import Imgdataset

num_epoch=10

train_address = "./myDIV2K"

train_dataset = Imgdataset(train_address)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
print(train_loader)
print("begin")

network=SRCNN()
optimizer = optim.Adam(network.parameters(), lr=0.01)
loss_fn=nn.MSELoss()

# sample=next(iter(train_loader))
# print(sample)
# img=sample[0]
# lab=sample[1]
# print(img.shape)
# print(lab.shape)


for epoch in range(num_epoch):
    print("epoch: ",epoch," ")
    total_loss=0
    total_correct=0
    for batch in train_loader:
        images,labels=batch
        print("images: ",images.shape)
        print("labels: ",labels.shape)

        preds=network(images)
        print("pres:",preds.shape)
        loss = loss_fn(preds,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss_item: ",loss.item())
        total_loss+=loss.item()

    print("total_loss:",total_loss)

torch.save(network,"./network.pth")