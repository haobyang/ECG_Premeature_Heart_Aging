import torch
import torchvision
from torchvision import models
from torchvision import transforms
from torch.autograd import Variable
import torch
import torch.nn as nn
from PIL import Image
import pylab as plt
import numpy as np
import cv2
import pickle
import matplotlib.gridspec as gridspec
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import copy
from sklearn import preprocessing
import os 
os.chdir("D:\\age\\04")




class Extractor():
    """ 
    when using pytorch, 
    we have to use hook to preserve the gradients. 
    """
    def __init__(self, model, target_layer = 'conv1'):
        self.model = model
        self.target_layer = target_layer # the layer you want to plot
        self.gradient = None

    def save_gradient(self, grad):
        self.gradient=grad

    def __call__(self, x):
        self.gradients = []

        # you have to revise this section
        # if the model structure is different from the resnet18 in resnet_multitask_se_1n.py
        ############################################################
        for name,module in self.model._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
            if name == self.target_layer:
                x.register_hook(self.save_gradient)
                target_activation=x

        x = self.model.avgpool(x)
        x=x.view(1,-1)
        x = self.model.fc(x)
        ##############################################################
        return target_activation, x




class GradCam():
    def __init__(self, model, target_layer_name, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = Extractor(self.model, target_layer_name)

    
    def __call__(self, input, size = (600,1000)):
        if self.cuda:
            target_activation, output = self.extractor(input.cuda())
        else:
            target_activation, output = self.extractor(input)


        one_hot = np.ones((1, output.size()[-1]), dtype = np.float32)
        one_hot = torch.tensor(one_hot)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = self.extractor.gradient.cpu().data.numpy()
         # shape（c, h, w）
        target = target_activation.cpu().data.numpy()[0]
        # shape（c,）
        weights = np.mean(grads_val, axis = (2, 3))[0, :]
        # the height and width of cam must be the same as target's
        cam = np.zeros(target.shape[1 : ], dtype = np.float32)
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        # values less than 0, replaced by 0
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, size)
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        return cam, output.item()



# the order of the leads
lead_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
#the index of lead you used to training the model
locate=0

#load model
import training as fb
from resnet_multitask_se_1n import resnet18
model = resnet18
model_name = 'res18_multi_se_1n_all'
demographic = False
multi = True


# load testing set
file_address = open('D:\\age\\03\\test_all_shuffle.pkl', 'rb')
data_all = pickle.load(file_address)
file_address.close()
len(data_all)

# plot activation map
# data_all : a list contains tuples including 12 leads, (lead data, age, agegp, gender)
# lead_name : the order of the lead
# locate : the index of lead you used to training the model
# model : the model
# model_name : the model name
# row_num : the number of the observation
# cut : the number of time steps you want to remove respectivly
# demographic: the paramter of eval_model() from training.py
# multi: the paramter of eval_model() from training.py
def plot_am(data_all, lead_name, locate,  model, model_name, row_num, cut, demographic, multi) :
    data = data_all[row_num]
    file_name = str(data[1])
    data = data[0]
    img = pd.DataFrame(data).T
    img.columns = lead_name

    # plot the single lead data
    dpi_value =300
    plt.figure(figsize= (25,3))
    signal = img[lead_name[locate]]
    plt.plot(signal, color='black', linewidth=3)
    plt.xlim((0,5000))
    plt.ylim((-20,20))
    plt.style.use('seaborn-white')

    # eliminate the margin
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())

    plt.savefig("./plot/lead{}_no{}_true{}.png".format(lead_name[locate],row_num,file_name), dpi=dpi_value,bbox_inches="tight")
    plt.close()

    # plot activation map
    data = torch.from_numpy(data[0,:])
    data = data.unsqueeze(dim = 0).unsqueeze(dim = 0).unsqueeze(dim = 0)
    data = Variable(data.cuda())
    data = data.type(torch.cuda.FloatTensor)

    # load model and weight
    net = model().cuda()
    pre_trained_model = torch.load('./'+model_name+'/Fold0_spatial.pkl')
    net.load_state_dict(pre_trained_model)

    size = (2500,300)
    layer = []
    for a in ['conv1',"layer1","layer2","layer3","layer4"]:
        target = a
        # grad cam is here!!!!
        grad_cam = GradCam(model = net, target_layer_name = target , use_cuda=True)
        mask, pred = grad_cam(data, size)
        heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap)/255

        # read the single lead just plotted previously
        img = cv2.imread("./plot/lead{}_no{}_true{}.png".format(lead_name[locate],row_num,file_name), 1) # 1 specifies to load a color image
        # resize the figure of single lead data
        img = np.float32(cv2.resize(img, size))/255
        # merge the figure with the heatmape produced by grad cam
        img = cv2.addWeighted(img,0.6,heatmap,0.4,0) 
        layer.append(img)


    # plot the importance for each time section
    BATCH = 1
    saved_path = "./" + model_name
    weight_path = saved_path + '/Fold0'

    mape_list = []
    s = cut
    for z in range(int(5000/s)): # ecah sample
        test = copy.deepcopy(data_all[row_num])
        test[0][:,(z*s):((z+1)*s)] = 0
        test = [test]
        MAPE = fb.eval_model_one(model,test,BATCH,weight_path,saved_path,[locate],demographic, multi)
        mape_list = mape_list + [MAPE]*s

    lead_important = pd.DataFrame(np.array([mape_list]).T, columns=['MAPE'])

    y_name = 'MAPE'
    lead_list = list(range(1,5001,1))
    score_list = lead_important.loc[:,y_name] 

    # re scale the importance
    score_list = score_list - np.min(score_list)
    scale = preprocessing.MinMaxScaler( feature_range=(0,1) ) 
    score_list = scale.fit_transform(np.array(score_list).reshape(-1, 1)) # Data

    plt.figure(figsize= (25,3))
    plt.bar(lead_list, score_list.squeeze(), width=1, label = 'lead_list')
    plt.xlim((0,5000))
    plt.ylim((0,1))
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())  
    plt.savefig('./plot/lead{}_no{}_step{}.png'.format(lead_name[locate],row_num,s), dpi=dpi_value,bbox_inches="tight")
    plt.close()

    # read the single lead just plotted previously
    img = cv2.imread("./plot/lead{}_no{}_true{}.png".format(lead_name[locate],row_num,file_name), 1) # 1 specifies to load a color image
    # read the bar chart of time step importance just plotted
    heatmap = cv2.imread('./plot/lead{}_no{}_step{}.png'.format(lead_name[locate],row_num,s), 1) # 1 specifies to load a color image
    # resize the plots
    img = np.float32(cv2.resize(img, size))/255
    heatmap = np.float32(cv2.resize(heatmap, size))/255
    img = cv2.addWeighted(img,0.6,heatmap,0.4,0) 
    # the heatmaps of each layer
    img = np.concatenate((layer[0],layer[1],layer[2],layer[3],layer[4], img), axis=0)
    cv2.imwrite("./plot/lead{}_no{}_true{}_pred{}.png".format(lead_name[locate],row_num,file_name,int(round(pred))), np.uint8(255*img))





plot_am(data_all, lead_name, locate,  model, model_name, 1, 20, demographic, multi) 



