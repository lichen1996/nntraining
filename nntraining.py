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
        if torch.cuda.is_available() == True:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor
        #loading data
        input_dim=inputx.shape[1]#  X.shape[1]
        h1_dim=4096#512
        h2_dim=2048
        h3_dim=128
        output_dim=outputy.shape[1] # Y.shape[1]
        learning_rate=5e-5
        self.data_num=inputx.shape[0]
        self.X = torch.from_numpy(inputx)
        self.Y = torch.from_numpy(outputy)


        #create net
        self.net = nn.Sequential(
            nn.Linear(input_dim, h1_dim),#7000 2048
            nn.ReLU(),
            nn.Linear(h1_dim, h2_dim),#2048 512
            nn.ReLU(),
            nn.Linear(h2_dim,h2_dim),#512  128
            # nn.ReLU(),
            # nn.Linear(h3_dim, h2_dim),#128 512
            nn.ReLU(),
            nn.Linear(h2_dim, output_dim) #512 3000
        )


        # lossfunction and opt
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate, weight_decay=1e-5) #


    #training
    # def training(self,epochs_num):
        epochs_num=1000
        losslist=[]
        iteration=range(epochs_num)
        for epoch in range(epochs_num):
            for trainNO in range(self.data_num):
                frameside=self.X[trainNO]
                frameside = Variable(frameside.type(self.dtype))
                frameinte=self.Y[trainNO]
                frameinte = Variable(frameinte.type(self.dtype))

                # forward
                self.net=self.net.cuda()
                out = self.net(frameside)
                loss = self.criterion(out, frameinte)
                # backward
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # losslist[epochs_num]=loss.data[0]
            # if (epoch + 1) % 1 == 0:
            print('Epoch[{}/{}], loss: {:.6f}'
                     .format(epoch + 1, epochs_num, loss.data[0]))
            losslist.append(float(loss.data[0]))

            if (epoch + 1) % 100 == 0:
                plt.figure(1)
                plt.ylim(0, 1)
                plt.xlim(3,epochs_num)
                plt.plot(iteration, losslist)
                plt.xlabel('Iteration')
                plt.ylabel('MSE')
                plt.title('Mean Squared Error')
                plt.savefig('i.png')
                plt.show()

        # file.write(str(losslist))
        torch.save(self.net, 'net4k.pkl')  # save net






