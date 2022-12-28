import pickle
import numpy as np
import torch
import pandas as pd

import os
os.chdir("/workspace/Desktop/age/04")
import training as fb

# parameters  
EPOCH = 3
BATCH = 20
LR = 0.001
STOP = 3
loss_function = 'mse + entropy'
gpu = 0

order = None
demographic = False
multi = True

from resnet_multitask_se import resnet18
from lstm import LSTM
model_name = 'test' 
model = resnet18


#########################################################

file_name = ['train','val','test']

for i in file_name:
    file_address = open(i+'_all_shuffle.pkl', 'rb')
    globals()[i] = pickle.load(file_address)
    file_address.close()


# small number of data for ensuring running
test = test[0:10]
val = val[0:40]
data = train[0:80]



# CV
saved_path = "./"+ model_name +"/"
FOLD = 3
train_fold = True
testing_set = None
# #########################
train_fold_loss, train_fold_mse, train_fold_mae, train_fold_mape, val_fold_loss, cv_loss, cv_mse, cv_mae, cv_mape, cv_acc, cv_pred, cv_pred2, cv_true = fb.train_model(model, data, val, EPOCH,BATCH,LR,STOP,saved_path,train_fold,FOLD,testing_set,order,demographic, multi, gpu)



# release memory
torch.cuda.empty_cache()


#training all
saved_path = "./" + model_name +"_all/"
FOLD = 1
train_fold = False
testing_set = test
#########################



train_fold_loss, train_fold_mse, train_fold_mae, train_fold_mape, val_fold_loss, cv_loss, cv_mse, cv_mae, cv_mape, cv_acc, cv_pred, cv_pred2, cv_true = fb.train_model(model, data, val, EPOCH,BATCH,LR,STOP,saved_path,train_fold,FOLD,testing_set,order,demographic,multi,gpu)






