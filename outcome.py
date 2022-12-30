import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
from glob import glob
from scipy import stats
import pandas as pd
from sklearn.metrics import r2_score
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from pandas_ml import ConfusionMatrix
from csv import reader
from matplotlib import pyplot as plt
import matplotlib
import scipy.stats
import seaborn as sns


import os
os.chdir("/workspace/Desktop/age/04")



def make_gp(x, cut = 6):
    if x > cut:
        return 1
    elif x < cut*(-1):
        return 2
    else:
        return 0


# output the result of hold out set 
def outcome(method):

    original_add = os.getcwd()
    os.chdir("./"+method+"/")

    # read the file in the specific folder 
    file_name = [i.split(os.getcwd()+'\\')[1].split('.pkl')[0] for i in glob(os.getcwd()+"/*.pkl")]
    #['cv_loss', 'cv_mae', 'cv_mape', 'cv_pred', 'cv_rmse', 'cv_true', 'train_fold_loss', 'train_fold_mae', 'train_fold_mape', 'train_fold_rmse', 'val_fold_loss']
    for i in file_name:
        file_address = open(i+'.pkl', 'rb')
        globals()[i] = pickle.load(file_address)
        file_address.close()

    # make prediction result become a dataframe
    # pred : the predicted heart age
    # true : the EKG_age
    # agegp : age group 
    # pred_to_class : discretize predicted heart age
    # d : the difference between the predicted value and the true values
    # d_ae : the absolute error of the difference
    # d_ape : the absolute percentage error of the difference
    # age_dfpg : discretize the difference
    out = pd.DataFrame()
    out['pred'] = cv_pred
    out['true'] = cv_true
    out['agegp'] = pd.cut(out['true'],bins = list(range(20,76,5))+[500],labels=np.arange(len(list(range(20,76,5))+[500])-1),right= True, include_lowest= True)
    out['pred_to_class'] = pd.cut(out['pred'],bins = [0]+list(range(25,76,5))+[500],labels=np.arange(len(list(range(20,76,5))+[500])-1),right= True, include_lowest= True)
    out['d'] = out.pred-out.true
    out['d_ae'] = abs(out.d)
    out['d_ape'] = abs((out.pred-out.true) / out.true) * 100
    out['age_dfgp'] = out.apply(lambda x : make_gp(x['d']), axis = 1) 

    os.chdir(original_add)
    return out



out = outcome('res18_multi_se_all_01_e25')
# check if there is any na
assert sum(out.isnull().sum())==0, 'there is something strange!'

# output and deliver the predicted age
file_address = open('D:\\age\\03\\test_all_df.pkl', 'rb')
findid = pickle.load(file_address)
file_address.close()

findid = findid[['req_no', 'patient_id','gender','EKG_age']]
# check the order of observations
assert sum(findid.EKG_age != out.true)==0,'the order is wrong!'

findid['predicted_age'] = out.pred
findid['age_difference'] = out.d
findid['age_dfgp'] = out.age_dfgp

findid.gender = findid.gender.replace(1,"M").replace(0,"F")
# change the column names
#list(findid)
#['req_no', 'patient_id', 'gender', 'EKG_age', 'predicted_age', 'age_difference', 'age_dfgp']
findid.columns = ['ReqNo','patientid_his','gender','EKG_age', 'predicted_age', 'age_difference','age_dfgp6']

# findid['age_dfgp9'] = findid.apply(lambda x : make_gp(x['age_difference'],9), axis = 1) 

# save as .csv
findid.to_csv("aging_20200713.csv", index = False)


# use the heatmap to see the relationship between age_dfg and agegp
dfv = pd.crosstab(out.age_dfgp, out.agegp)
sns.heatmap(dfv, cmap='YlOrRd', annot=False)
plt.savefig("age_dfg_agegp_0713.png",bbox_inches="tight")
plt.close()

# plot the confusion matrix between the discretized predicted age and true age group 
confusion_matrix = ConfusionMatrix(out.agegp, out.pred_to_class)
font = {'size': 14}
matplotlib.rc('font', **font)
confusion_matrix.plot(normalized=True)
plt.savefig("confusion.png",bbox_inches="tight")
plt.close()
# classification detail of each group
cm = ConfusionMatrix(out.agegp.astype(int), out.pred_to_class.astype(int))
cm.print_stats()



# whether there is a signigicant difference between the performance of proposed method and mayo clinic
#mae
dat1 = outcome("res18_multi_se_all_01_e25")['d_ae']
dat2 = outcome("mayo_all")['d_ae']

# check if there is na
assert (dat1.isnull().sum())==0, 'there is na!!'
assert (dat2.isnull().sum())==0, 'there is na!!'

# test if the distribution is normal distribution
# if >0.05, it is normal distribution
scipy.stats.shapiro(dat1)[1]
scipy.stats.shapiro(dat2)[1]

# test if the variances are the same
# if >0.05,  s1==s2, they are from the same population
scipy.stats.levene(dat1, dat2, center = 'median')[1]

# test if the mean values are the same (t test)
# if <0.05, reject h0, the mean values are different
scipy.stats.ttest_ind(dat1, dat2, equal_var = True)[1]


# mape
dat1 = outcome("res18_multi_se_all_01_e25")['d_ape']
dat2 = outcome("mayo_all")['d_ape']

# check if there is na
assert (dat1.isnull().sum())==0, 'there is na!!'
assert (dat2.isnull().sum())==0, 'there is na!!'

# test if the distribution is normal distribution
scipy.stats.shapiro(dat1)[1]
scipy.stats.shapiro(dat2)[1]

# test if the variances are the same
scipy.stats.levene(dat1, dat2, center = 'median')[1]

# test if the mean values are the same (t test)
scipy.stats.ttest_ind(dat1, dat2, equal_var = False)[1]



# Cumulative Distribution of Difference Between Predicted and Actual Age
dpi_value =300
font = {'size': 10}
matplotlib.rc('font', **font)
fig = plt.figure(figsize= (10,4))
fig.subplots_adjust(top=0.85)
gs1 = gridspec.GridSpec(1,2)
gs1.update(wspace=0.2, hspace=0) # set the spacing between axes. 

ax = plt.subplot(gs1[0])
ax = sns.distplot(out.d,bins=12, kde=True, hist=False, kde_kws=dict(cumulative=True))
data_x, data_y = ax.lines[0].get_data()
xi = 6  # coordinate where to find the value of kde curve
yi = np.interp(xi,data_x, data_y)
ax.plot([xi],[yi], marker="o")
ax.annotate('6.0 yrs ({}%)'.format(round(yi*100,1)), (xi+2,yi-0.05))
ax.set_ylabel('Cumulative percent (%)', fontsize = 14)
xi = 0  # coordinate where to find the value of kde curve
yi = np.interp(xi,data_x, data_y)
ax.plot([xi],[yi], marker="o")
ax.annotate('0.0 yrs ({}%)'.format(round(yi*100,1)), (xi+2,yi-0.05))
ax.set_yticks( np.arange(0.0, 1.2, 0.2))
ax.set_yticklabels(list(range(0,120,20)))
ax.set_xlabel('Year', fontsize = 14)

ax = plt.subplot(gs1[1])
ax = sns.distplot(out.d,bins=list(range(-40,50,5)), kde=False, hist_kws={"ec":"k"})
ax.set_xticks(list(range(-20,30,10)))
ax.set_xlabel('Year', fontsize = 14)
ax.set_ylabel('Count', fontsize = 14)

fig.suptitle("Difference Between Predicted and Actual Age", fontsize=18)
plt.savefig('difference_dist.png', dpi=dpi_value,bbox_inches="tight")
plt.close()




# plot true and predicted values
fig = plt.figure(figsize=(3,3), dpi=300)
fig.subplots_adjust(top=0.9)
#plt.tight_layout()
#plt.plot(out.true,out.pred, marker="o")
plt.scatter(out.true,out.pred, alpha=0.6, s= 1 )
plt.plot([15,85],[15,85],color = 'red')
# plt.xticks(list(range(0,len(val_fold_loss[0])+2,2)),list(range(1,len(val_fold_loss[0])+2,2)))
plt.xlabel('Chronological Age')
plt.ylabel('Predicted Age')
plt.text(15, 80,'Pearson\'s correlation : '+str(round(pearsonr(out.true,out.pred)[0],2)),fontsize = 8)#, horizontalalignment='right', verticalalignment='top')
plt.text(15, 75,'R-squared : '+str(round(r2_score(out.true, out.pred),2)),fontsize = 8)#, horizontalalignment='right', verticalalignment='top')
fig.suptitle('Predicted Age and Chronological Age',fontsize = 12)
plt.savefig("actual_predicted_scatter.png",bbox_inches="tight")
plt.close()



  
# result barchart
# compare the model with gender and without gender
# load the result record
data = pd.read_csv("result03_bar.csv",header=0)

# set up for the plot
x_labels = ['5-fold CV','testing data']
acc  = ['fold_mean_mae', 'test_mae']
ka = ['fold_mean_mape', 'test_mape']
dpi_value =300
fig = plt.figure(figsize=(8,6))
width = 0.5 # width of bar
length = 2
x = np.arange(length)*1.5
font = {'size':16}
matplotlib.rc('font', **font)

# plot two figures together
# the first one (MAE)
ax = plt.subplot(1, 2, 1)

# without gender
ax.bar(x, data.loc[data.model=='res18_multi_se',acc].values.squeeze(), 
        width, label='without gender', capsize=5, color = "orangered",
        yerr=data.loc[data.model=='res18_multi_se','fold_sd_mae'].values.tolist()+[0])

# with gender
ax.bar(x + width, data.loc[data.model=='run_res18_multi_se_gender_b',acc].values.squeeze(), 
        width, label='with gender', capsize=5, color = "darkgreen",
        yerr=data.loc[data.model=='run_res18_multi_se_gender_b','fold_sd_mae'].values.tolist()+[0])

ax.set_ylabel('MAE', fontsize = 16)
ax.set_ylim(5.5,6.5)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
plt.xticks(rotation=45)
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)


# plot two figures together
# the second one (MAPE)
ax = plt.subplot(1, 2, 2)

# without gender
ax.bar(x, data.loc[data.model=='res18_multi_se',ka].values.squeeze(), 
        width, label='without gender', capsize=5, color = "orangered",
        yerr=data.loc[data.model=='res18_multi_se','fold_sd_mape'].values.tolist()+[0])

# with gender
ax.bar(x + width, data.loc[data.model=='run_res18_multi_se_gender_b',ka].values.squeeze(), 
        width, label='with gender', capsize=5, color = "darkgreen",
        yerr=data.loc[data.model=='run_res18_multi_se_gender_b','fold_sd_mape'].values.tolist()+[0])

ax.set_ylabel('MAPE', fontsize = 16)
ax.set_ylim(14,16)
ax.set_xticks(x)
ax.set_xticklabels(x_labels)
ax.legend()
plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
fig.tight_layout()
plt.xticks(rotation=45)
plt.style.use('seaborn-white')
plt.savefig('compare.png', dpi=dpi_value)
plt.close()




# use the testing set to plot lead importance
lead_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# which model you want to test
import training as fb
from resnet_multitask_se import resnet18
model = resnet18
demographic = False
multi = True
model_name = 'res18_multi_se_all_01_e25'

BATCH = 512
saved_path = "./" + model_name 
weight_path = saved_path + '/Fold0'

# y_name : ['Loss','RMSE','MAE','MAPE','ACC'] choose one 
# remove : whether test the importance by removing single lead respectivly
# model : the model you want to test
# demographic: the paramter of eval_model() from training.py
# multi: the paramter of eval_model() from training.py
def lead_importance(y_name, remove, model, demographic, multi) :
    loss_list = []
    rmse_list = []
    mae_list = []
    mape_list = []
    acc_list = []
    for k in range(12): # each lead
        file_address = open('/workspace/Desktop/age/03/test_all_shuffle.pkl', 'rb')
        test = pickle.load(file_address)
        file_address.close()
        print('The amount of dataset | {} \nLead {}..............'.format(len(test),lead_name[k]))

        for z in range(len(test)): # ecah sample
            if remove:
                test[z][0][k,:] = 0 #delete one lead
            else:
                kk = list(range(0,k,1))+list(range(k+1,12,1))
                test[z][0][kk,:] = 0 #keep one lead

        LOSS, RMSE, MAE, MAPE, ACC, _, _, _ = fb.eval_model(model,test,BATCH,weight_path,saved_path,demographic=demographic,multi=multi)

        loss_list.append(LOSS)
        rmse_list.append(RMSE)
        mae_list.append(MAE)
        mape_list.append(MAPE)
        acc_list.append(ACC)   


    lead_important = pd.DataFrame(np.array([loss_list,rmse_list,mae_list,mape_list,acc_list]), index=['Loss','RMSE','MAE','MAPE','ACC'], columns=lead_name)


    lead_list = list(lead_important)
    score_list = lead_important.loc[y_name,:]
    dpi_value = 300
    plt.figure(figsize=(1600/dpi_value, 1000/dpi_value), dpi=dpi_value)
    plt.bar(lead_list, score_list, width=0.8, label = 'lead_list')
    for i, v in enumerate(score_list):
        plt.text(i-0.4, v+ 0.01 , str(v)[0:5], size=8)

    plt.title("Importance of Each Lead")
    plot_title = "Removed Lead" if remove else "Remained Lead"
    plt.xlabel(plot_title)
    plt.ylabel(y_name)
    plt.savefig('./lead_importance_{}_{}_{}.png'.format(y_name,model_name,remove), dpi=dpi_value,bbox_inches="tight")
    plt.close()


lead_importance("MAPE", True, model, demographic, multi)
lead_importance("MAPE", False, model, demographic, multi)