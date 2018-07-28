import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from torch.utils.data import DataLoader



#inputx and outputy must be np form
#X and Y are 2dim array,every line of the X matches the corresponding line of Y,which is groud truth
#after running the code, the net will be store into file "net.pkl"



class frame_net(nn.Module):
    def __init__(self,inputx,outputy):
        super(frame_net,self).__init__()

        #loading data
        input_dim=inputx.shape[1]#  X.shape[1]
        h1_dim=512
        h2_dim=128
        # h3_dim=64
        output_dim=outputy.shape[1] # Y.shape[1]
        learning_rate=1e-3
        self.data_num=inputx.shape[0]
        self.X = torch.from_numpy(inputx)
        self.Y = torch.from_numpy(outputy)


        #create net
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),
            nn.ReLU(),
            nn.Linear(h2_dim, h1_dim),
            nn.ReLU(),
            nn.Linear(h1_dim, output_dim)
        )

        # lossfunction and opt
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5) #


    #training
    # def training(self,epochs_num):
        epochs_num=100
        losslist=[]
        for epoch in range(epochs_num):
            for trainNO in range(self.data_num):
                frameside=self.X[trainNO]
                frameside = Variable(frameside.type(torch.FloatTensor))
                frameinte=self.Y[trainNO]
                frameinte = Variable(frameinte.type(torch.FloatTensor))

                # forward
                out = self.net(frameside)
                loss = self.criterion(out, frameinte)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # losslist[epochs_num]=loss.data[0]
            if (epoch + 1) % 2 == 0:
               print('Epoch[{}/{}], loss: {:.6f}'
                     .format(epoch + 1, epochs_num, loss.data[0]))
        torch.save(self.net, 'net.pkl')  # save net






