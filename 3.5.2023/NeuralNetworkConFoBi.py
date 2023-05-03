import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import time


def sigmoid(x):
    return  1 /( 1 + (math.e)**(-1 * x))

def sigmoid_deriviate(x):
    a = sigmoid(x)
    a = np.reshape(a,(-1,1))
    b = 1 - sigmoid(x)
    b = np.reshape(b,(-1,1))
    b = np.transpose(b)
    return np.diag(np.diag(np.matmul(a,b)))


split_ratio = 0.5
eta = 0.5
epochs = 100

data = pd.read_csv('Dataset.csv',header=None, delimiter=';')
data = np.array(data)



for i in range(np.shape(data)[1]):

    min = np.min(data[:,i])
    max = np.max(data[:,i])
    for j in range(np.shape(data)[0]):
        data[j,i] = (data[j,i] - min) / (max - min)

split_line_number = int(np.shape(data)[0] * split_ratio)
x_train = data[:split_line_number,:-1]
x_test = data[split_line_number:,:-1]
y_train = data[:split_line_number,-1]
y_test = data[split_line_number:,-1]


input_dimension = np.shape(x_train)[1]
l1_neurons = 9
l2_neurons = 1


w1 = np.random.uniform(low=-1,high=1,size=(input_dimension,l1_neurons))
w2 = np.random.uniform(low=-1,high=1,size=(l1_neurons,l2_neurons))

MSE_train = []
MSE_test = []

for i in range(epochs):


    sqr_err_epoch_train = []
    sqr_err_epoch_test = []

    output_train = []
    output_test = []

    for j in range(np.shape(x_train)[0]):
        # Feed-Forward

            # Layer 1

        net1 = np.matmul(x_train[j],w1)
        o1 = sigmoid(net1)
        o1 = np.reshape(o1,(-1,1))


            # Layer 2
        net2 = np.matmul(np.transpose(o1),w2)
        o2 = net2


        output_train.append(o2[0])

        # Error
        err = y_train[j] - o2[0]
        sqr_err_epoch_train.append(err**2)

        # Back propagation
        f_driviate = sigmoid_deriviate(net1)
        w2_f_deriviate = np.matmul(f_driviate,w2)
        w2_f_deriviate_x = np.matmul(w2_f_deriviate,np.transpose(np.reshape(x_train[j],(-1,1))))
        w1 = np.subtract(w1 , np.transpose((eta * err * -1 * 1 * w2_f_deriviate_x)))
        w2 = np.subtract(w2 , (eta * err * -1 * 1 * o1))

    mse_epoch_train = 0.5 * ((sum(sqr_err_epoch_train))/np.shape(x_train)[0])
    MSE_train.append(mse_epoch_train)

    for j in range(np.shape(x_test)[0]):
        # Feed-Forward

        # Layer 1
        net1 = np.matmul(x_test[j], w1)
        o1 = sigmoid(net1)
        o1 = np.reshape(o1, (-1, 1))

        # Layer 2
        net2 = np.matmul(np.transpose(o1), w2)
        o2 = net2

        output_test.append(o2[0])

        # Error
        err = y_test[j] - o2[0]
        sqr_err_epoch_test.append(err ** 2)

    mse_epoch_test = 0.5 * ((sum(sqr_err_epoch_test))/np.shape(x_test)[0])
    MSE_test.append(mse_epoch_test)


    # Poly fits

        # Train
    m_train , b_train = np.polyfit(y_train,output_train,1)

        # Test
    m_test , b_test = np.polyfit(y_test, output_test, 1)

    print('R^2 Train: ', m_train,b_train,'R^2 Test:',m_test,b_test)

    # Plots
    fig, axs = plt.subplots(3, 2, figsize=(7,8))
    plt.tight_layout(pad=2)
    axs[0, 0].plot(MSE_train,'b')
    axs[0, 0].set_title('MSE Train')
    axs[0, 1].plot(MSE_test,'r')
    axs[0, 1].set_title('Mse Test')

    axs[1, 0].plot(y_train, 'b')
    axs[1, 0].plot(output_train,'r')
    axs[1, 0].set_title('Output Train')
    axs[1, 1].plot(y_test, 'b')
    axs[1, 1].plot(output_test,'r')
    axs[1, 1].set_title('Output Test')

    axs[2, 0].plot(y_train, output_train, 'b.')
    axs[2, 0].plot(y_train, m_train*y_train+b_train,'r')
    axs[2, 0].set_title('Regression Train')
    axs[2, 1].plot(y_test, output_test, 'b.')
    axs[2, 1].plot(y_test,m_test*y_test+b_test,'r')
    axs[2, 1].set_title('Regression Test')
    if i == (epochs - 1):
        plt.savefig('Results.jpg')
    plt.show()
    time.sleep(1)
    plt.close(fig)


