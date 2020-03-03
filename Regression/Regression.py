#This code is a very simple example of Regression.

import math

import torch
import numpy as np
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.autograd import Variable
import matplotlib.pyplot as plt

x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]]))
y_data = Variable(torch.Tensor([[2.0], [4.0], [6.0]]))
plt.figure(figsize=(10,4))
plt.scatter(x_data.data.numpy(), y_data.data.numpy(), color = "orange")
plt.title('Regression Analysis')
plt.xlabel('Independent varible')
plt.ylabel('Dependent varible')
plt.show()

class LinearRegressionModel(Module):

    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearRegressionModel()
criterion = torch.nn.MSELoss(size_average = False)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)
fig, ax = plt.subplots(figsize=(12,7))

for epoch in range(500):
    pred_y = model(x_data)
    loss = criterion(pred_y, y_data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # plot and show learning process
    plt.cla()
    ax.set_title('Regression Analysis', fontsize=35)
    ax.set_xlabel('Independent variable', fontsize=24)
    ax.set_ylabel('Dependent variable', fontsize=24)
    ax.set_xlim(-1.05, 1.5)
    ax.set_ylim(-0.25, 1.25)
    ax.scatter(x_data.data.numpy(), y_data.data.numpy(), color="orange")
    ax.plot(x_data.data.numpy(), pred_y.data.numpy(), 'g-', lw=3)
    ax.text(1.0, 0.1, 'Step = %d' % epoch, fontdict={'size': 24, 'color': 'red'})
    ax.text(1.0, 0, 'Loss = %.4f' % loss.data.numpy(),
            fontdict={'size': 24, 'color': 'red'})
    fig.canvas.draw()


new_var = Variable(torch.Tensor([[4.0]]))
pred_y = model(new_var)
print("predict (after training)", 4, model(new_var).data[0][0])
