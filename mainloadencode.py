import os
from nntraining import frame_net
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import torch
import matplotlib.pyplot as plt
import re


trainpath = '../encode/train/'
testpath = '../encode/test/'

if __name__ == '__main__':
    X = []
    Y = []
    data = []
    temp = []
    directory = os.fsencode(trainpath)
    files = os.listdir(directory)
    # print(files.type())
    sortfile = sorted(files, key=lambda x: int(re.sub(b'\D', b'', x)))
    for file in sortfile:
        filename = os.fsdecode(file)
        #print(filename)
        frame = int(filename[1:])
        prefix = os.fsdecode(directory)
        if frame % 10 == 0:
            data.append(np.loadtxt(prefix + filename).flatten())

    for i in range(len(data)-2):
        del temp[:]
        temp.append(data[i])
        temp.append(data[i+2])
        temp = np.array(temp)
        X.append(temp.flatten())
        Y.append(data[i+1])
        temp = temp.tolist()

    #get test data
    testX = []
    testY = []
    testdata = []
    directory = os.fsencode(testpath)
    files = os.listdir(directory)
    sortfile = sorted(files, key=lambda x: int(re.sub(b'\D', b'', x)))
    for file in sortfile:
        filename = os.fsdecode(file)
        prefix = os.fsdecode(directory)
        if filename.startswith('U'):
            testdata.append(np.loadtxt(prefix + filename).flatten())

    for i in range(len(testdata)-2):
        del temp[:]
        temp.append(testdata[i])
        temp.append(testdata[i + 2])
        temp = np.array(temp)
        testX.append(temp.flatten())
        testY.append(testdata[i+1])
        temp = temp.tolist()

    X = np.asarray(X)
    #X.reshape(X, (X.shape, X[0].shape))
    Y = np.asarray(Y)
    #Y.reshape(Y, (Y.shape, Y[0].shape))

    testX = np.asarray(testX)
    #testX.reshape(testX, (testX.shape, testX[0].shape))
    testY = np.asarray(testY)

    #training
    model = frame_net(X, Y)
    # model.training()

    #plot
    trainednet = torch.load('net.pkl')
    testX = torch.from_numpy(testX)
    for testnum in range(9):
        truedata = testY[testnum]
        inputdata = testX[testnum]

        inputdata = Variable(inputdata.type(torch.FloatTensor))
        prediction = trainednet(inputdata)
        predictionnp = prediction.data.numpy()

        # CVAE_data = CVAE_data[0]
        plt.figure(1)
        plt.scatter(predictionnp, truedata, c='b')
        # plt.scatter(truedata[0::2], truedata[1::2], c='r', marker='x')
        plt.legend(('Prediction', 'Ground Truth'))
        plt.title('N.O.: ' + str(testnum))
        plt.savefig('../image/l2l/test/' + str(testnum) + '.png')
        plt.close()

    # CNN_data = model.generate(inputdata)

    # criterion = torch.nn.MSELoss()
    # los = criterion(CNN_data, truedata)
    # print(los.data[0])

    # plt.figure(1)
    # plt.scatter(CNN_data.data.numpy(), truedata.data.numpy(), c='b')
    # plt.show()
