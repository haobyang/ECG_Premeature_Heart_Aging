import pandas as pd 
from sqlalchemy import create_engine
import os 
from collections import Counter
from biosppy.signals import ecg
import numpy as np
from struct import pack, unpack
import zlib
import pickle
from sklearn.utils import shuffle
import time
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import enum
import hashlib
from sklearn.model_selection import GroupShuffleSplit
import seaborn as sns

os.chdir("/workspace/Desktop/age/04")

from const import LEAD_NAMES, LEAD_LENGTH, LEAD_SAMPLING_RATE, LEAD_DROP_LEN


def unpack_leads(pack_lead_data):
    f_lead_data = unpack('d' * LEAD_LENGTH * len(LEAD_NAMES), zlib.decompress(pack_lead_data))
    assert len(f_lead_data) == LEAD_LENGTH * len(LEAD_NAMES)
    ret = {}
    for lead_id, lead_name in enumerate(LEAD_NAMES):
        data_loc = lead_id * LEAD_LENGTH
        ret[lead_name] = list(f_lead_data[data_loc:data_loc + LEAD_LENGTH])
    return ret


# Preprocess for lead data to remove baseline wander and high freq noise
# lead : single lead data
def preprocess_leads(lead):
    LEAD_SAMPLING_RATE = 500   
    corrected_signal, _, _ = ecg.st.filter_signal(lead, 'butter', 'highpass', 2,1, LEAD_SAMPLING_RATE)
    preprocessed_signal, _, _ = ecg.st.filter_signal(corrected_signal, 'butter', 'lowpass', 12,35, LEAD_SAMPLING_RATE)
    return preprocessed_signal


# cut the time steps of each lead
# length : how many time steps you want to keep
# signals : single lead data
def cut_signal(signals, length):
    assert len(signals) >= length, "length is too long!"
    return signals[0:length]


# the key to merge lead data (x) and label (y)
# deidentify the req_no and patient id 
def get_mhash(req_no, patient_id, salt= "#"):
    s = hashlib.sha256()
    s.update(f"{req_no}{salt}{patient_id}".encode("utf-8"))
    return s.hexdigest()



# load lead data (.db)
address = "/workspace/Desktop/ecg_db/ecg_data_20190519_20191230.db/"
engine = create_engine(f'sqlite:///{address}') 
table_names = engine.table_names() # the table name in the db
print(table_names) # print all the names of the tables
lead  = pd.read_sql('ECGLeads', engine)  # read the table as dataframe
len(lead) # 735819 whole data


# load labels (y)
label = pd.read_csv("Healthy_ECG_20200620.csv", header= 0) # age data
# label = pd.read_csv("LVH_NCTU20190814.csv", header= 0) # LVH data
# label = pd.read_csv("ECG_HFMI_20200620.csv", header= 0) # MI HF data

# drop duplicated row
label = label.drop_duplicates()
# create the key (mhash)
label['mhash'] = label.apply(lambda x: get_mhash(x['ReqNo'], x['patientid_his']),axis = 1)


# label.dtypes
# label.patientid_his.astype('object')

# merge lead data and age data
lead = lead.merge(label, on = 'mhash' ,how = 'inner').reset_index(drop=True) #left_on=[], right_on=[]
# check if there is any na
assert sum(lead.isnull().sum())==0, lead.isnull().sum()

# check if there are some duplicated keys
assert len(np.unique(lead.mhash))==len(lead), "there are some duplicated keys"
#check before drop
assert sum(lead.req_no != lead.ReqNo)==0, 'req_no from two dataframe are different'
assert sum(lead.patient_id.astype('int64') != lead.patientid_his)==0, "patient id from two dataframe are different"
lead = lead.drop(['ReqNo', 'patientid_his'], axis=1)


# about heartbeat, refer to ECGDL.preprocess.transform import RandomSelectHB
# valid ecg : taint_code <300 and heartbeat_cnt>=8
lead = lead[(lead.taint_code <300) & (lead.heartbeat_cnt>=8)].reset_index(drop=True) 


# take age >=20 because of IRB
lead = lead[(lead.EKG_age>=20)].reset_index(drop=True) 


# create age group, 5 years a group
lead['agegp'] = pd.cut(lead['EKG_age'],bins = range(20,121,5),labels=np.arange(len(range(20,121,5))-1),right= True, include_lowest= True)
lead['agegp'].value_counts()

# change F to 0, M to 1
lead.gender = lead.gender.replace("F",0).replace("M",1)


# MI, MI2 (OMI), HF
# lead['gender'] = 999
# mi = lead[lead.dis_MI_e==1].reset_index(drop=True) #6392
# mi2 = lead[lead.dis_OMI_e==1].reset_index(drop=True) #2808
# hf = lead[lead.dis_HF_e==1].reset_index(drop=True) #20617

# LVH
# lvh = lead[lead.LVH==1].reset_index(drop=True) 


#check patient with multiple ecg
print('number of patient with multiple ECGs:', sum(lead.patient_id.value_counts()>1)) 
print('the maximum number of ECGs a patient has:', max(lead.patient_id.value_counts()))

# patients with multiple ECGs
overlap = lead[lead.patient_id.isin(lead.patient_id.value_counts()[lead.patient_id.value_counts()>1].index.tolist())].reset_index(drop=True)
# patients with single ECGs
lead = lead[lead.patient_id.isin(lead.patient_id.value_counts()[lead.patient_id.value_counts()==1].index.tolist())].reset_index(drop=True)


# divided into training testing validaiton sets
# the ratio for each set is 8:2:2
# stratify by 'gender' 'agegp'
# for patients only have one record
train, test, _ , _ = train_test_split(lead, lead['agegp'], test_size=(1/6), stratify = lead[['gender','agegp']],random_state=42, shuffle = True)
train, val, _ , _ = train_test_split(train, train['agegp'], test_size=(1/5), stratify = train[['gender','agegp']],random_state=42, shuffle = True)

train = train.reset_index(drop=True)
test = test.reset_index(drop=True)
val = val.reset_index(drop=True)


# divided into training testing validaiton sets
# for patient only have multiple records
gss = GroupShuffleSplit(n_splits=1, test_size=(1/6), random_state=42)
for train_idx, test_idx in gss.split(X=overlap, groups=overlap.patient_id):
    test_over = overlap.iloc[test_idx,].reset_index(drop=True)
    train_over = overlap.iloc[train_idx,].reset_index(drop=True)

gss = GroupShuffleSplit(n_splits=1, test_size=(1/5), random_state=42)
for train_idx, val_idx in gss.split(X=train_over, groups=train_over.patient_id):
    val_over = train_over.iloc[val_idx,].reset_index(drop=True)
    train_over = train_over.iloc[train_idx,].reset_index(drop=True)


# concat and shuffle dataset
# patients with one ecg + patients with multiple ecgs
train = shuffle(pd.concat([train, train_over], ignore_index=True, axis= 0),random_state=123).reset_index(drop=True)
test = shuffle(pd.concat([test, test_over], ignore_index=True, axis= 0),random_state=123).reset_index(drop=True)
val = shuffle(pd.concat([val, val_over], ignore_index=True, axis= 0),random_state=123).reset_index(drop=True)

# see the ratio of each age group
round(pd.crosstab(train['agegp'], train['gender'])/len(train),3)
round(pd.crosstab(val['agegp'], val['gender'])/len(val),3)
round(pd.crosstab(test['agegp'], test['gender'])/len(test),3)

round(train['agegp'].value_counts()/len(train),3)
round(test['agegp'].value_counts()/len(test),3)
round(val['agegp'].value_counts()/len(val),3)

# check no overlap patientid in each set
assert sum(train.patient_id.isin(np.unique(test.patient_id).tolist())) == 0, 'Cross-contamination!!!'
assert sum(train.patient_id.isin(np.unique(val.patient_id).tolist())) == 0, 'Cross-contamination!!!'
assert sum(val.patient_id.isin(np.unique(test.patient_id).tolist())) == 0, 'Cross-contamination!!'

# re group people above 75 together
train['agegp'] = pd.cut(train['EKG_age'],bins = list(range(20,76,5))+[500],labels=np.arange(len(list(range(20,76,5))+[500])-1),right= True, include_lowest= True)
test['agegp'] = pd.cut(test['EKG_age'],bins = list(range(20,76,5))+[500],labels=np.arange(len(list(range(20,76,5))+[500])-1),right= True, include_lowest= True)
val['agegp'] = pd.cut(val['EKG_age'],bins = list(range(20,76,5))+[500],labels=np.arange(len(list(range(20,76,5))+[500])-1),right= True, include_lowest= True)


# plot and count each age group 
# train : a dataframe with column 'agegp'
# name : the name to save the plot as a file
def count_agegp(train, name):
    x_labels=['20 to 25','26 to 30','31 to 35','36 to 40','41 to 45','46 to 50','51 to 55','56 to 60','61 to 65','66 to 70','71 to 75','above 75']
    lead_list = train['agegp'].value_counts(sort=False).index
    score_list = train['agegp'].value_counts(sort=False)
    dpi_value = 300
    plt.figure(figsize=(1600/dpi_value, 1000/dpi_value), dpi=dpi_value)
    plt.bar(lead_list, score_list, width=0.8, label = 'lead_list')
    for i, v in enumerate(score_list):
        plt.text(i-0.4, v+ 0.01 ,v, size=8)

    plt.xticks([i-0.5 for i in lead_list],x_labels,rotation=45)
    plt.title("The Number of Each Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Count")
    plt.savefig('./{}.png'.format(name), dpi=dpi_value,bbox_inches="tight")
    plt.close()

count_agegp(train,'train_agegp')
count_agegp(test,'test_agegp')
count_agegp(val,'val_agegp')


# bar chart
# plot the distribution of EKG_age 
# draw : a dataframe with column 'EKG_age'
# name : the name to save the plot as a file
def age_dist (draw, name):
    dpi_value = 300
    plt.figure(figsize=(1000/dpi_value, 1000/dpi_value), dpi=dpi_value)
    ax = sns.distplot(draw['EKG_age'],bins=10, kde=False, hist_kws={"ec":"k"})
    ax.set_ylabel('Count', fontsize = 14)
    ax.set_xlabel('Age', fontsize = 14)

    # # # Creating another Y axis
    second_ax = ax.twinx()
    # #Plotting kde without hist on the second Y axis
    sns.distplot(draw['EKG_age'], ax=second_ax, kde=True, hist=False)
    second_ax.set_yticks([])
    plt.title(name, fontsize=22)
    plt.savefig("{}_distribution.png".format(name),dpi=dpi_value,bbox_inches="tight")
    plt.close()


age_dist(train,'train')
age_dist(test,'test')
age_dist(val,'val')




# become tuples (lead data, age, agegp, gender) #testing validation
# become tuples (patient id, lead data, age, agegp, gender) #training 
# dataset : a dataframe contains patientid, lead data, EKG_age, agegp, gender
# file_name : a name for the tuple list to save as pkl.
# training : whether this dataset is training data
# lead_name : the order of 12 leads you want to save
# cut_len : how many time steps you want to keep

# the order of lead data
lead_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def become_tuple(dataset, file_name, training, lead_name=lead_name, cut_len = 5000):
    lead = dataset
    print('number of rows in dataframe:', len(lead))
    lead_list = []
    for j in range(len(lead)):
        data = lead.iloc[j,]
        data = pd.DataFrame(unpack_leads(data.lead_data), columns = lead_name)
        data = data.apply(lambda x: preprocess_leads(cut_signal(x,cut_len)), axis = 0)
        data = data.T.to_numpy()
        if training:
            lead_list.append((lead.patient_id[j],data,lead.EKG_age[j],lead.agegp[j],lead.gender[j]))
        else :
            lead_list.append((data,lead.EKG_age[j],lead.agegp[j],lead.gender[j]))
    print('number of tuples in list:', len(lead_list))
    print('Now saving {}..........'.format(file_name+'_shuffle.pkl'))
    write = open(file_name+'_shuffle.pkl', 'wb')
    pickle.dump(lead_list, write)
    write.close()  


become_tuple(train, 'train_all', True)
become_tuple(test, 'test_all', False)
become_tuple(val, 'val_all', False)


# save the testing set as dataframe in order to deliver the outcome
# ['req_no', 'patient_id','gender','EKG_age']
write = open('test_all_df.pkl', 'wb')
pickle.dump(test[['req_no', 'patient_id','gender','EKG_age']], write)
write.close()  

