import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F
import pickle
from tqdm import tqdm 
import time
import numpy as np
from sklearn.model_selection import GroupKFold
import os


# record the training and evaluaiton process in a file
#info : the information (string) you want to record
#path : the path of the file you want to save the record into
def info_record(info,path):
    text = open(path + ".txt", "a")
    text.write(info+'\n')
    text.close()
    print(info)


# GroupKFold
# group by patient id
# data : training data
# FOLD : number of folds you want to split
# n : this is fold n
def train_test_fold(data, FOLD, n): 
    group = np.array([i[0] for i in data]) # extract patient id as the group information
    group_kfold = GroupKFold(n_splits=FOLD)
    data = np.array([i[1:5] for i in data])
    for fold_number, (train_index, test_index) in enumerate(group_kfold.split(X=data, groups=group)):
        if n > fold_number:
            continue
        elif n == fold_number: # this is fold n   
            train_load, cv_load = data[train_index].tolist(), data[test_index].tolist()
            break
    return train_load, cv_load




# calculate the sum of absolute percentage error
# y_true : the actual EKG age
# y_pred : the predicted age from the model
def mape(y_true, y_pred):
    return sum(np.abs((y_true - y_pred) / y_true)) * 100



# normalize each lead respectively
# data_list : training data or testing data
# training : True if the data_list is training data
# train_mean : can not be None if the data_list is not training data
# train_sd : the same as train_mean
def normal_each(data_list, training = True, train_mean=None, train_sd= None):
    if training:
        print('\ncalculate mean')
        for i in tqdm(range(len(data_list)),ascii=True): 
            if i==0:
                d1 = data_list[i][1]
                train_mean = np.sum(d1, axis=1)
                continue
            d1 = data_list[i][1]
            d1 = np.sum(d1, axis=1)
            train_mean = train_mean+d1

        train_mean = (train_mean/(len(data_list*5000))).reshape(12,1)

        print('\ncalculate std')
        for i in tqdm(range(len(data_list)),ascii=True): 
            if i==0:
                d1 = data_list[i][1]
                train_sd = np.sum((d1-train_mean)**2, axis=1)
                continue
            d1 = data_list[i][1]
            d1 = np.sum((d1-train_mean)**2, axis=1)
            train_sd = train_sd+d1

        train_sd = ((train_sd/(len(data_list*5000)-1))**0.5).reshape(12,1)

        print('\nfor training')
        new_list = []
        for i in tqdm(range(len(data_list)),ascii=True): 
            d1 = data_list[i][1]
            d1 = (d1-train_mean)/train_sd
            new_list.append((data_list[i][0],d1,data_list[i][2],data_list[i][3],data_list[i][4]))
        print('\ntraining finish')
        return new_list, train_mean, train_sd


    else:
        print('\nfor testing/validating')
        train_mean = train_mean
        train_sd = train_sd

        assert (train_mean.shape)==(12,1), 'invalid train_mean!!!'
        assert (train_sd.shape)==(12,1), 'invalid train_std!!!'

        new_list = []
        for i in tqdm(range(len(data_list)),ascii=True): 
            d1 = data_list[i][0]
            d1 = (d1-train_mean)/train_sd
            new_list.append((d1,data_list[i][1],data_list[i][2],data_list[i][3]))
        print('\ntesting/validating finish')
        return new_list



# normalize the whole dataset
# data_list : training data or testing data
# training : True if the data_list is training data
# train_mean : can not be None if the data_list is not training data
# train_sd : the same as train_mean
def normal_all(data_list, training = True, train_mean=None, train_sd= None):
    if training:
        print('\ncalculate mean')
        for i in tqdm(range(len(data_list)),ascii=True): 
            if i==0:
                d1 = data_list[i][1]
                train_mean = np.sum(d1)
                continue
            d1 = data_list[i][1]
            d1 = np.sum(d1)
            train_mean = train_mean+d1

        train_mean = (train_mean/(len(data_list*5000*12)))

        print('\ncalculate std')
        for i in tqdm(range(len(data_list)),ascii=True): 
            if i==0:
                d1 = data_list[i][1]
                train_sd = np.sum((d1-train_mean)**2)
                continue
            d1 = data_list[i][1]
            d1 = np.sum((d1-train_mean)**2)
            train_sd = train_sd+d1

        train_sd = ((train_sd/(len(data_list*5000*12)-1))**0.5)

        print('\nfor training')
        new_list = []
        for i in tqdm(range(len(data_list)),ascii=True): 
            d1 = data_list[i][1]
            d1 = (d1-train_mean)/train_sd
            new_list.append((data_list[i][0],d1,data_list[i][2],data_list[i][3],data_list[i][4]))
        print('\ntraining finish')
        return new_list, train_mean, train_sd


    else:
        print('\nfor testing/validating')
        train_mean = train_mean
        train_sd = train_sd

        assert len([train_mean])==1, 'no train_mean!!!'
        assert len([train_sd])==1, 'no train_std!!!'

        new_list = []
        for i in tqdm(range(len(data_list)),ascii=True): 
            d1 = data_list[i][0]
            d1 = (d1-train_mean)/train_sd
            new_list.append((d1,data_list[i][1],data_list[i][2],data_list[i][3]))
        print('\ntesting/validating finish')
        return new_list




# model : the model structure which you want to build
# data : the training dataset
# val : the dataset used for early stop
# EPOCH : the number of epoch
# BATCH : the batch size
# LR : the learning rate
# STOP : stop training based on the  
# saved_path : where you want to save the model weights
# train_fold : whether to implement the CV
# FOLD : k-fold CV, k is the FOLD
# testing_set : the testing dataset which is a list of tuples
# order : the order of the leads, please input a list
# demographic : whether to use other demographic features
# multi : is this model use multitask learning
# gpu : which gpu you want to use
def train_model(model, data, val, EPOCH,BATCH,LR,STOP,saved_path,train_fold, FOLD, testing_set = None, order = None, demographic= False, multi= True, gpu = 0):
    # metric for every epoch of folds during training
    train_fold_loss = []
    train_fold_rmse = []
    train_fold_mae = []
    train_fold_mape = []
    train_fold_acc = []

    # early stop
    val_fold_loss = [] 

    # metric for each fold/testing set during evaluation
    cv_loss = []
    cv_rmse = []
    cv_mae = []
    cv_mape = []
    cv_acc = []

    # predicted and actual values
    cv_pred = []
    cv_pred2 = []
    cv_true = []
    
    for n in range(FOLD): #k-fold
        info_record('\nStart to train.................................\n', saved_path.split('/')[-2])
        
        # Build model
        spatial =  model().cuda(gpu)
        model_param = spatial.parameters()

        # optimizor and loss function
        optim = torch.optim.Adam(model_param, lr = LR) 
        #loss_func = nn.SmoothL1Loss()
        loss_func = nn.MSELoss()
        loss_func2 = nn.CrossEntropyLoss()


        if train_fold: # implement CV
            train_load, cv_load = train_test_fold(data, FOLD, n)
            train_load = DataLoader(train_load, shuffle=False, batch_size = BATCH)
            info_record('5-fold training data amount {}'.format(len(train_load.dataset)), saved_path.split('/')[-2])

        else :        
            # train all training set and evaluate the testing set
            data = np.array([i[1:5] for i in data]).tolist()
            train_load = DataLoader(data, shuffle=False, batch_size = BATCH)
            cv_load =  testing_set
            info_record('all training data amount {}'.format(len(train_load.dataset)), saved_path.split('/')[-2])

        # metric for each epoch during training
        train_loss = []
        train_rmse = []
        train_mae = []
        train_mape = []
        train_acc = []

        # early stop test in every epoch
        val_loss = []
        val_min = None
        best_epoch = 0
        val_count = 0

        # start to train fold n
        for epoch in range(EPOCH):
            start_time = time.time()
            info_record('\nFOLD {} | EPOCH {}..............'.format(n,epoch), saved_path.split('/')[-2])

            LOSS = 0
            RMSE = 0
            MAE = 0
            MAPE = 0
            ACC = 0

            #training mode
            spatial.train()

            for step, (x,y, gp, gender) in enumerate(train_load):
                
                # change the order of 12 leads
                # pick specific number of lead
                if order is not None:
                    x = x[:,order,:]
                
                # load training data
                x,y = Variable(x.cuda(gpu)), Variable(y.cuda(gpu).type(torch.cuda.FloatTensor))
                x = x.unsqueeze(dim = 1).type(torch.cuda.FloatTensor)

                # put the data into the model
                if demographic:
                    # apply the demographic features
                    # only for multitask
                    gender = Variable(gender.cuda(gpu))
                    gp = Variable(gp.cuda(gpu))
                    result, result2 = spatial(x, gender)
                else :
                    if multi: # with multitask
                        gp = Variable(gp.cuda(gpu))
                        result, result2 = spatial(x)
                    else:     # without multitask
                        result = spatial(x)
                
                # calculate the loss and do the backpropagation
                loss1 = loss_func(result.squeeze(),y)
                loss2 = loss_func2(result2,gp) if multi else None
                loss = loss1+loss2 if multi else loss1
                # print(loss1, loss2, loss)
                optim.zero_grad()   
                loss.backward()         
                optim.step()        
             
                #metric
                #LOSS = LOSS + F.smooth_l1_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy()
                if multi:
                    LOSS = LOSS + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy() + F.cross_entropy(result2,gp, reduction='sum').cpu().detach().numpy()
                    ACC = ACC + result2.cpu().detach().max(1)[1].eq(gp.cpu().detach()).sum().item()
                else:
                    LOSS = LOSS + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy() 
                RMSE = RMSE + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy()
                MAE = MAE + F.l1_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy()
                MAPE = MAPE + mape(y.cpu().detach().numpy(), result.squeeze().cpu().detach().numpy())

            
            #metric for each epoch during training
            train_loss.append(LOSS/len(train_load.dataset))
            train_rmse.append((RMSE/len(train_load.dataset))**0.5)
            train_mae.append(MAE/len(train_load.dataset))
            train_mape.append(MAPE/len(train_load.dataset))
            train_acc.append(ACC/len(train_load.dataset))

            info_record('TRAIN loss {}| rmse {} | mae {} | mape {} | acc {}'.format(LOSS/len(train_load.dataset),(RMSE/len(train_load.dataset))**0.5,MAE/len(train_load.dataset),MAPE/len(train_load.dataset),ACC/len(train_load.dataset)), saved_path.split('/')[-2])

            # early stop
            val_load = DataLoader(val, shuffle=False, batch_size = BATCH)

            # evaluation mode
            spatial.eval()        
            with torch.no_grad():
                LOSS = 0
                for step, (x,y, gp , gender ) in enumerate(val_load):  
                    # change the order of 12 leads
                    # pick specific number of lead
                    if order is not None:
                        x = x[:,order,:] 
                    
                    x,y = Variable(x.cuda(gpu)), Variable(y.cuda(gpu).type(torch.cuda.FloatTensor))
                    x = x.unsqueeze(dim = 1).type(torch.cuda.FloatTensor)        


                    # put the data into the model
                    if demographic:
                        # apply the demographic features
                        # only for multitask
                        gender = Variable(gender.cuda(gpu))
                        gp = Variable(gp.cuda(gpu))
                        result, result2 = spatial(x, gender)
                    else :
                        if multi: # with multitask
                            gp = Variable(gp.cuda(gpu))
                            result, result2 = spatial(x)
                        else:     # without multitask
                            result = spatial(x)
                
                    
                    #LOSS = LOSS + F.smooth_l1_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy()
                    if multi:
                        LOSS = LOSS + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy() + F.cross_entropy(result2,gp, reduction='sum').cpu().detach().numpy()
                    else:
                        LOSS = LOSS + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy() 



                val_loss.append(LOSS/len(val_load.dataset))

                if epoch > 0 : 
                    if (LOSS/len(val_load.dataset)) < val_min:
                        val_min = LOSS/len(val_load.dataset)
                        val_count = 0
                        best_epoch = epoch
                        info_record('VAL loss {} | early stop count {} |  ...............saving model.....'.format((LOSS/len(val_load.dataset)), val_count), saved_path.split('/')[-2])

                        # save the best weight until now
                        weight_path = saved_path + 'Fold'+str(n) 
                        torch.save(spatial.state_dict(), weight_path + '_spatial'+'.pkl')
                    else :
                        if epoch > 24:
                            val_count = val_count +1 # at least train 25 epoches
                        info_record('VAL loss {} | early stop count {}'.format((LOSS/len(val_load.dataset)), val_count), saved_path.split('/')[-2])
                    
                    if val_count >= STOP: # stop training 
                        info_record('Fold {} | early stop on epoch {}'.format(n,epoch), saved_path.split('/')[-2])
                        info_record('The best model is on epoch {} !!!'.format(best_epoch), saved_path.split('/')[-2])
                        break

                    if epoch == (EPOCH-1) : # does not meet the criterion of early stop
                        info_record('\n============ Fold {} | at final epoch {} ============='.format(n,epoch), saved_path.split('/')[-2])

                else :
                    val_min = LOSS/len(val_load.dataset)
                    val_count = 0
                    best_epoch = epoch

                    info_record('VAL loss {} | early stop count {} |  ...............saving model.....'.format((LOSS/len(val_load.dataset)), val_count), saved_path.split('/')[-2])
                    weight_path = saved_path + 'Fold'+str(n) 
                    torch.save(spatial.state_dict(), weight_path + '_spatial'+'.pkl')    

            # calculate the time each epoch cost
            end_time = time.time()
            info_record('\n======== EPOCH SPEND TIME {} min ========'.format(round((end_time-start_time)/60,2)), saved_path.split('/')[-2])
            

        # evaluate the fold (k-fold CV) when train_fold == TRUE
        # evaluate the hold out testing set when train_fold == False
        LOSS, RMSE, MAE, MAPE, ACC, predict, predict2, actual = eval_model(model, cv_load,BATCH,weight_path,saved_path,order,demographic,multi,gpu)

        info_record('CV loss {}| rmse {} | mae {} | mape {} | acc {} '.format(LOSS,RMSE,MAE,MAPE,ACC), saved_path.split('/')[-2])

        # metric for evaluation
        cv_loss.append(LOSS)
        cv_rmse.append(RMSE)
        cv_mae.append(MAE)
        cv_mape.append(MAPE)
        cv_acc.append(ACC)

        cv_pred = cv_pred + predict 
        cv_pred2 = cv_pred2 + predict2 
        cv_true = cv_true + actual

        
        #metric during training
        train_fold_loss.append(train_loss)
        train_fold_rmse.append(train_rmse)
        train_fold_mae.append(train_mae)
        train_fold_mape.append(train_mape)
        train_fold_acc.append(train_acc)
        val_fold_loss.append(val_loss)

    return train_fold_loss, train_fold_rmse, train_fold_mae, train_fold_mape, val_fold_loss, cv_loss, cv_rmse, cv_mae, cv_mape, cv_acc, cv_pred, cv_pred2, cv_true





# cv_load : the testing set used to evaluate the well-trained model
# weight_path : the directory of the best model weights
# saved_path : the directory to save the .txt for recording the process
def eval_model(model, cv_load,BATCH,weight_path,saved_path, order = None, demographic= False, multi=True, gpu=0):
    cv_load = DataLoader(cv_load, shuffle=False, batch_size = BATCH)
    info_record('\nStart to evaluate.................................\n', saved_path.split('/')[-2])
    info_record('Evaluation data amount {}'.format(len(cv_load.dataset)), saved_path.split('/')[-2])

    # load well-trained model
    spatial =  model().cuda(gpu)
    info_record('loading spatial model...........{}\n'.format(weight_path + '_spatial'+'.pkl'), saved_path.split('/')[-2])
    state_dict_path = torch.load(weight_path + '_spatial'+'.pkl')
    spatial.load_state_dict(state_dict_path)

    # evaluation mode
    spatial.eval()        
    with torch.no_grad():
        LOSS = 0
        RMSE = 0
        MAE = 0
        MAPE = 0
        ACC = 0
        predict = []
        predict2 = []
        actual = []
        
        for step, (x,y, gp , gender) in enumerate(cv_load):
            # change the order of 12 leads
            # pick specific number of lead
            if order is not None:
                x = x[:,order,:]

            x,y = Variable(x.cuda(gpu)), Variable(y.cuda(gpu).type(torch.cuda.FloatTensor))
            x = x.unsqueeze(dim = 1).type(torch.cuda.FloatTensor)

            # put the data into the model
            if demographic:
                # apply the demographic features
                # only for multitask
                gender = Variable(gender.cuda(gpu))
                gp = Variable(gp.cuda(gpu))
                result, result2 = spatial(x, gender)
            else :
                if multi: # with multitask
                    gp = Variable(gp.cuda(gpu))
                    result, result2 = spatial(x)
                else:     # without multitask
                    result = spatial(x)
            

            predict = predict + result.cpu().detach().numpy().squeeze().tolist()
            predict2 = predict2 + result2.cpu().detach().max(1)[1].numpy().squeeze().tolist() if multi else predict2
            actual = actual + y.cpu().detach().numpy().tolist()

            #LOSS = LOSS + F.smooth_l1_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy()
            if multi:
                LOSS = LOSS + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy() + F.cross_entropy(result2,gp, reduction='sum').cpu().detach().numpy()
                ACC = ACC + result2.cpu().detach().max(1)[1].eq(gp.cpu().detach()).sum().item()
            else:
                LOSS = LOSS + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy() 
            RMSE = RMSE + F.mse_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy()
            MAE = MAE + F.l1_loss(result.squeeze(),y,reduction='sum').cpu().detach().numpy()
            MAPE = MAPE + mape(y.cpu().detach().numpy(), result.squeeze().cpu().detach().numpy())

    return (LOSS/len(cv_load.dataset)), (RMSE/len(cv_load.dataset))**0.5 , (MAE/len(cv_load.dataset)), (MAPE/len(cv_load.dataset)), (ACC/len(cv_load.dataset)), predict, predict2, actual



# when batch size (BATCH) is 1
def eval_model_one(model, cv_load,BATCH,weight_path,saved_path, order = None, demographic= False, multi=True, gpu=0):
    cv_load = DataLoader(cv_load, shuffle=False, batch_size = BATCH)

    # load well-trained model
    spatial =  model().cuda(gpu)
    state_dict_path = torch.load(weight_path + '_spatial'+'.pkl')
    spatial.load_state_dict(state_dict_path)

    # evaluation mode
    spatial.eval()        
    with torch.no_grad():
        LOSS = 0
        RMSE = 0
        MAE = 0
        MAPE = 0
        ACC = 0
        predict = []
        predict2 = []
        actual = []
        
        for step, (x,y, gp , gender) in enumerate(cv_load):
            # change the order of 12 leads
            # pick specific number of lead
            if order is not None:
                x = x[:,order,:]

            x,y = Variable(x.cuda(gpu)), Variable(y.cuda(gpu).type(torch.cuda.FloatTensor))
            x = x.unsqueeze(dim = 1).type(torch.cuda.FloatTensor)

            # put the data into the model
            if demographic:
                # apply the demographic features
                # only for multitask
                gender = Variable(gender.cuda(gpu))
                gp = Variable(gp.cuda(gpu))
                result, result2 = spatial(x, gender)
            else :
                if multi: # with multitask
                    gp = Variable(gp.cuda(gpu))
                    result, result2 = spatial(x)
                else:     # without multitask
                    result = spatial(x)
            MAPE = MAPE + mape(y.cpu().detach().numpy(), result.squeeze(dim=0).cpu().detach().numpy())
    return (MAPE/len(cv_load.dataset))



    


