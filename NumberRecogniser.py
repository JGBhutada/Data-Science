import torch
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

#GETING THE DATASET
train = datasets.MNIST("", train = True, download=False, transform = transforms.Compose([transforms.ToTensor()]))
test = datasets.MNIST("", train = False, download=False, transform = transforms.Compose([transforms.ToTensor()]))

#CONVERTING THE DATASET INTO BATCHES OF SIZE 10
trainset = torch.utils.data.DataLoader(train, batch_size = 10, shuffle = True)
testset = torch.utils.data.DataLoader(test, batch_size = 10, shuffle = True)

for data in trainset:
    X, y = data
    break


import torch.nn as nn
import torch.nn.functional as F

#CREATING THE ARCHITECTURE OF OUR NEURAL NET (784*64*64*64*10)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28*28, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = (self.fc4(x))
        return F.log_softmax(x, dim = 1)
        return x
net = Net()
#print(net)


x = torch.rand(28,28)
x = x.view(-1, 28*28)
output = net(x)

#TRAINING OVER TRAINING DATASET
import torch.optim as optim
optimizer = optim.Adam(net.parameters(), lr = 0.01)
EPOCHS= 3
for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X.view(-1, 28*28))
        loss = F.nll_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)

#TESTING THE DATASET AND CALCULATING THE ACCURACY
correct = 0
total = 0
with torch.no_grad():
    for data in testset:
        X,y = data
        output = net(X.view(-1, 28*28))
        for idx, i in enumerate(output):
            if torch.argmax(i) == y[idx]:
                correct +=1
            total +=1
print("Accuracy = ", correct/total)

plt.imshow(X[0].view(28,28))
plt.show()
print(torch.argmax(net(X[0].view(-1, 784))[0]))


