import torch
from torch import nn
from torch import optim
from model import SRCNN
import torch.nn.functional as F
from dataloader import Imgdataset
from vgg_extract import extract_feature


num_epoch=10

train_address = "./myDIV2K"

train_dataset = Imgdataset(train_address)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=100,shuffle=True)
print(train_loader)
print("begin")

network=torch.load('./myvggnetwork_1.pth')
optimizer = optim.Adam(network.parameters(), lr=0.01)
loss_fn=nn.MSELoss()
# loss_fn2=nn.L1Loss()

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
        loss = 0.001*loss_fn(extract_feature(preds),extract_feature(labels))+loss_fn(preds,labels)
        print("loss_calculate_over!")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print("loss_item: ",loss.item())
        total_loss+=loss.item()
    print("total_loss:",total_loss)
    torch.save(network, "./myvggnetwork_1.pth")

