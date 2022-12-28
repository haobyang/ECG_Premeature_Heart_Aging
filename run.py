import pickle
import os
import pandas as pd
import numpy as np
import torch

# os.chdir("/workspace/Desktop/age/04")
os.chdir("/home/chinyu25/age/04")
import training as fb

# parameters  
EPOCH = 1000
BATCH = 680
LR = 0.01
STOP = 10
loss_function = 'mse + entropy'
gpu = 0

order = None
demographic = False
multi = True

from resnet_multitask_se3 import resnet18
model_name = 'res18_multi_se3' #build the folder first
model = resnet18

######################################################

# load dataset
file_name = ['train_all','val_all','test_all']  # the whole dataset
obj_name = ['data','val','test']

for i , j in zip(file_name,obj_name):
    file_address = open(i+'_shuffle.pkl', 'rb')
    globals()[j] = pickle.load(file_address)
    file_address.close()
    fb.info_record('The amount of dataset | {} : {}'.format(j, len(globals()[j])),model_name)


# small number of data for ensuring running
# test = test[0:40]
# val = val[0:40]
# data = data[0:100]

training_size = len(data)

# CV
saved_path = "./"+ model_name +"/"
FOLD = 5
train_fold = True
testing_set = None

train_fold_loss, train_fold_rmse, train_fold_mae, train_fold_mape, val_fold_loss, cv_loss, cv_rmse, cv_mae, cv_mape, cv_acc, cv_pred, cv_pred2, cv_true = fb.train_model(model, data, val, EPOCH,BATCH,LR,STOP,saved_path,train_fold,FOLD,testing_set, order, demographic, multi, gpu)


# save result
result_address =  saved_path 
file_name = ['train_fold_loss', 'train_fold_rmse', 'train_fold_mae', 'train_fold_mape', 'val_fold_loss', 'cv_loss', 'cv_rmse', 'cv_mae', 'cv_mape',  'cv_acc', 'cv_pred', 'cv_pred2', 'cv_true']
result = [ train_fold_loss, train_fold_rmse, train_fold_mae, train_fold_mape, val_fold_loss, cv_loss, cv_rmse, cv_mae, cv_mape, cv_acc, cv_pred, cv_pred2, cv_true]
for i in range(len(result)):
    add = open(result_address + file_name[i] + '.pkl', 'wb')
    pickle.dump(result[i], add)
    add.close()


# record the result
keep = {
    'model': model_name,
    'fold_mean_mae': round(np.mean(cv_mae),2),
    'fold_sd_mae' : round(np.std(cv_mae,ddof=1),2),
    'test_mae' : None,
    'fold_mean_mape' : round(np.mean(cv_mape),2),
    'fold_sd_mape' : round(np.std(cv_mape,ddof=1),2),
    'test_mape' : None,
    'training_size' : training_size,
    'loss_function' : loss_function,
    'count' : STOP,
    'batch' : BATCH,
    'lr' : LR
}


# release memory
torch.cuda.empty_cache()


# training all
saved_path = "./" + model_name +"_all/"
FOLD = 1
train_fold = False
testing_set = test

train_fold_loss, train_fold_rmse, train_fold_mae, train_fold_mape, val_fold_loss, cv_loss, cv_rmse, cv_mae, cv_mape, cv_acc, cv_pred, cv_pred2, cv_true = fb.train_model(model, data, val, EPOCH,BATCH,LR,STOP,saved_path,train_fold,FOLD,testing_set,order,demographic,multi, gpu)


# save result
result_address =  saved_path 
file_name = ['train_fold_loss', 'train_fold_rmse', 'train_fold_mae', 'train_fold_mape', 'val_fold_loss', 'cv_loss', 'cv_rmse', 'cv_mae', 'cv_mape',  'cv_acc', 'cv_pred', 'cv_pred2', 'cv_true']
result = [ train_fold_loss, train_fold_rmse, train_fold_mae, train_fold_mape, val_fold_loss, cv_loss, cv_rmse, cv_mae, cv_mape, cv_acc, cv_pred, cv_pred2, cv_true]
for i in range(len(result)):
    add = open(result_address + file_name[i] + '.pkl', 'wb')
    pickle.dump(result[i], add)
    add.close()




#record the result
keep['test_mae'] = round(np.mean(cv_mae),2)
keep['test_mape'] = round(np.mean(cv_mape),2)

result = pd.read_csv("result04.csv", header = 0)
result = result.append(pd.DataFrame(keep,index=[0],columns = result.columns), ignore_index = True) 

result.to_csv("result04.csv", index = False)