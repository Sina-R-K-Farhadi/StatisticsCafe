import torch
import torch.nn as nn
import torch.nn.functional as f
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


#torch.manual_seed(4)


class ConFoBiNet(nn.Module):
    def __init__(self, input_size, firstlayer_neurons=6):
        super().__init__()
        self.weights1 = nn.Linear(input_size, firstlayer_neurons, dtype=float)
        self.weights2 = nn.Linear(firstlayer_neurons, 1, dtype=float)

    def forward(self, x):
        x = f.sigmoid(self.weights1(x))
        x = self.weights2(x)
        return x


split_ratio = 0.8
epochs = 100


data = pd.read_csv('Dataset.csv',header=None, delimiter=';')
data = np.array(data)



for i in range(np.shape(data)[1]):

    min = np.min(data[:,i])
    max = np.max(data[:,i])
    for j in range(np.shape(data)[0]):
        data[j,i] = (data[j,i] - min) / (max - min)

split_line_number = int(np.shape(data)[0] * split_ratio)
data = torch.tensor(data)


x_train = data[:split_line_number,:-1]
x_test = data[split_line_number:,:-1]
y_train = data[:split_line_number,-1]
y_test = data[split_line_number:,-1]


input_dimension = np.shape(x_train)[1]
l1_neurons = 9

net = ConFoBiNet(input_dimension, l1_neurons)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.2, momentum=0)


Epochsloss_train = []
Epochsloss_test = []

for epoch in range(epochs):

    # Train
    epochloss_train = 0
    outputs_train = []
    for i in range(np.shape(x_train)[0]):
        optimizer.zero_grad()
        o = net(x_train[i,:])
        loss = criterion(o[0], y_train[i])
        epochloss_train += loss.item()
        loss.backward()
        optimizer.step()
        outputs_train.append(o.detach().numpy())

    epochloss_train = epochloss_train / np.shape(x_train)[0]
    Epochsloss_train.append(epochloss_train)


    # Test
    epochloss_test = 0
    outputs_test = []
    for i in range(np.shape(x_test)[0]):
        optimizer.zero_grad()
        o = net(x_test[i, :])
        loss = criterion(o[0], y_test[i])
        epochloss_test += loss.item()
        outputs_test.append(o.detach().numpy())

    epochloss_test = epochloss_test / np.shape(x_test)[0]
    Epochsloss_test.append(epochloss_test)

    # Poly fits

    # Train
    m_train, b_train = np.polyfit(y_train.detach().numpy(), outputs_train, 1)

    # Test
    m_test, b_test = np.polyfit(y_test.detach().numpy(), outputs_test, 1)

    print('R^2 Train: ', m_train,b_train,'R^2 Test:',m_test,b_test)

    # Plots
    fig, axs = plt.subplots(3, 2, figsize=(7,8))
    plt.tight_layout(pad=2)
    axs[0, 0].plot(Epochsloss_train, 'blue')
    axs[0, 0].set_title('MSE Train')
    axs[0, 1].plot(Epochsloss_test, 'red')
    axs[0, 1].set_title('Mse Test')

    axs[1, 0].plot(y_train, 'b')
    axs[1, 0].plot(outputs_train, 'r')
    axs[1, 0].set_title('Output Train')
    axs[1, 1].plot(y_test, 'b')
    axs[1, 1].plot(outputs_test, 'r')
    axs[1, 1].set_title('Output Test')

    axs[2, 0].plot(y_train, outputs_train, 'b.')
    axs[2, 0].plot(y_train, m_train[0] * y_train + b_train, 'r')
    axs[2, 0].set_title('Regression Train')
    axs[2, 1].plot(y_test, outputs_test, 'b.')
    axs[2, 1].plot(y_test, m_test[0] * y_test + b_test, 'r')
    axs[2, 1].set_title('Regression Test')
    if epoch == (epochs - 1):
        plt.savefig('Results.jpg')
    plt.show()
    time.sleep(1)
    plt.close(fig)


