import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, inputt=[1,16,16,32,32,64,64,64], output=[16,16,32,32,64,64,64,64], kernel=[7,5,5,5,5,3,3,3], pool=[2,4,2,4,2,2,2,2]):
        super(CNN,self).__init__()

        self.conv0 = nn.Sequential(
                nn.Conv2d(in_channels = inputt[0],
                        out_channels = output[0],
                        kernel_size = (1,kernel[0]),
                        padding = (0,60)),
                        #stride = 1,
                        #padding = 1),  
                nn.BatchNorm2d(output[0]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size= (1,pool[0]), stride=(1,2)) #, padding=1) 
        )
        self.conv1 = self._layer(inputt[1],output[1],kernel[1],pool[1])
        self.conv2 = self._layer(inputt[2],output[2],kernel[2],pool[2])
        self.conv3 = self._layer(inputt[3],output[3],kernel[3],pool[3])
        self.conv4 = self._layer(inputt[4],output[4],kernel[4],pool[4])
        self.conv5 = self._layer(inputt[5],output[5],kernel[5],pool[5])
        self.conv6 = self._layer(inputt[6],output[6],kernel[6],pool[6])
        self.conv7 = self._layer(inputt[7],output[7],kernel[7],pool[7])

        self.conv_s = nn.Sequential(
                nn.Conv2d(in_channels = 64,
                        out_channels = 128,
                        kernel_size = (12,1)),
                        #stride = 1,
                        #padding = 1),  
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size= (1,2))# , padding=1) 
        )

        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.5)

        self.fc1 = nn.Linear(1024,128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128,64)
        self.bn2 = nn.BatchNorm1d(64)


        self.out = nn.Linear(64,1)


    def _layer(self, i,o,k,p):
        return nn.Sequential(
                        nn.Conv2d(in_channels = i,
                               out_channels = o,
                               kernel_size = (1,k)),
                        nn.BatchNorm2d(o),
                        nn.ReLU(inplace=True),
                        nn.MaxPool2d(kernel_size= (1,p), stride=(1,2)))

                      

    
    def forward(self,x):
        #print('input shape {}'.format(x.shape))
        x = self.conv0(x)
        #print('conv0 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv1(x)
        #print('conv1 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv2(x)
        #print('conv2 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv3(x)
        #print('conv3 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv4(x)
        #print('conv4 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv5(x)
        #print('conv5 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv6(x)
        #print('conv6 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv7(x)
        #print('conv7 shape {}'.format(x.shape)) # 20, 64, 3, 157
        x = self.conv_s(x)
        #print('conv_s shape {}'.format(x.shape)) # 20, 64, 3, 157

        x = x.view(x.size(0), -1)
        #print('flatten shape {}'.format(x.shape)) # 20, 512

        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        #print('fc1 shape {}'.format(x.shape))
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        #print('fc2 shape {}'.format(x.shape))


        x = self.out(x)
        # print('out shape {}'.format(x.shape))
        return x
