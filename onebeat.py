import pandas as pd 
import os 
from biosppy.signals import ecg
import numpy as np
import pickle
from sklearn.utils import shuffle
import random
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib
import seaborn as sns


os.chdir("/workspace/Desktop/age/04")
from const import LEAD_NAMES, LEAD_LENGTH, LEAD_SAMPLING_RATE, LEAD_DROP_LEN


# cut heartbeat 
file_address = open('test_all_shuffle.pkl', 'rb')
lead = pickle.load(file_address)
file_address.close()    
print(len(lead))

# a well-preprocessed sample 
data = lead[0][0]


# the order of the lead
lead_name = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


# count the number of valid hearbeat which is at least 0.6 sec
# the location of R peak
# data : lead II  
# before : the duration (sec) before R peak
# after : the duraiton (sec) after R peak
def heartbeat_cnt(data, before=0.2, after = 0.4):
    # Use Lead II to get the R-peaks
    # data
    # r peak location
    rpeak = ecg.hamilton_segmenter(data, sampling_rate=LEAD_SAMPLING_RATE)[0]
    # each heartbeat is 0.2+0.4 sec
    _, rpeak = ecg.extract_heartbeats(data, rpeak, LEAD_SAMPLING_RATE,before=before, after=after)
    return len(rpeak), rpeak


heartbeat_cnt(data[1])



# locate : the index of lead II, use Lead II to get the R-peaks
# window : whether to circle each heartbeat
# circle each hearbeat
def circle_onebeat(data, save_name, locate = None, window = True, before = 0.2, after=0.4):
    dpi_value =300
    plt.figure(figsize= (6,10))
    gs1 = gridspec.GridSpec(12,1)
    gs1.update(wspace=0, hspace=0.3) # set the spacing between axes. 
    font = {'size': 8}
    matplotlib.rc('font', **font)
    for i in range(len(lead_name)):
        ax = plt.subplot(gs1[i])
        ax.plot(data[i], color='black', linewidth=0.6)
        ax.set_ylabel(lead_name[i])
        ax.set_yticks([])
        ax.set_xticks([])
        if window:
            count = 1
            _, peak = heartbeat_cnt(data[locate],before=before,after=after)
            for j in peak:
                if count % 2 == 0:
                    ax.plot([j+after*500,j+after*500],[min(data[i]),max(data[i])], color = 'red', linestyle='-', linewidth=0.5)
                    ax.plot([j-before*500,j-before*500],[min(data[i]),max(data[i])], color = 'red', linestyle='-', linewidth=0.5)
                    ax.plot([j+after*500,j-before*500],[min(data[i]),min(data[i])], color = 'red', linestyle='-', linewidth=0.5)
                    ax.plot([j+after*500,j-before*500],[max(data[i]),max(data[i])], color = 'red', linestyle='-', linewidth=0.5)
                    count+=1
                else :
                    ax.plot([j+after*500,j+after*500],[min(data[i]),max(data[i])], color = 'blue', linestyle='-', linewidth=0.5)
                    ax.plot([j-before*500,j-before*500],[min(data[i]),max(data[i])], color = 'blue', linestyle='-', linewidth=0.5)                    
                    ax.plot([j+after*500,j-before*500],[min(data[i]),min(data[i])], color = 'blue', linestyle='-', linewidth=0.5)
                    ax.plot([j+after*500,j-before*500],[max(data[i]),max(data[i])], color = 'blue', linestyle='-', linewidth=0.5)
                    count+=1
    plt.style.use('seaborn-white')
    plt.savefig("{}.png".format(save_name), dpi=dpi_value,bbox_inches="tight")
    plt.close()


circle_onebeat(data, "circle_one", locate= 1)
circle_onebeat(data, "circle_one2", window=False, locate= 1)




# split one sample (5000 time steps) into multiple observations
# which contain only one single heartbeat
# file_name : the name of the original .pkl which contains tuple list including data of 5000 time steps
# locate : the index of lead II
# train : whethter it is training set
# before : the duration before R peak
# after : the duration after R peak
def split_beat(file_name, locate , train, before=0.2, after=0.4):
    page = 0 # the label to specify which observations are from the same sample
    file_address = open(file_name+'.pkl', 'rb')
    lead = pickle.load(file_address)
    file_address.close()    
    print(len(lead))
    print(lead[0])
    lead_list = []

    for k in list(range(len(lead))):
        data = lead[k][1] if train else lead[k][0]

        # Use Lead II to get the R-peaks
        # data[1]
        # ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
        # r peak location
        rpeak = ecg.hamilton_segmenter(data[locate], sampling_rate=LEAD_SAMPLING_RATE)[0]

        lead_data_split = []
        for i in data:
            # each heartbeat 0.2+0.4 sec
            template, rpeak = ecg.extract_heartbeats(i, rpeak, LEAD_SAMPLING_RATE,before, after)
            lead_data_split.append(template)

        lead_data = np.array(lead_data_split)
        assert lead_data.shape[0] == 12 , "forget clearing up lead_data_split = []?"
        assert lead_data.shape[1] == len(rpeak) , "# of rpeak is strange."
        assert lead_data.shape[2] ==  int(before*LEAD_SAMPLING_RATE+after*LEAD_SAMPLING_RATE) , "each heartbeat (0.2+0.4 sec)*500hz = 300?"

        for j in list(range(len(rpeak))):
            if train:
                lead_list.append((lead[k][0],lead_data[:,j,:],lead[k][2],lead[k][3],lead[k][4],int(page)))
            else :
                lead_list.append((lead_data[:,j,:],lead[k][1],lead[k][2],lead[k][3],int(page)))
        
        page +=1

    print(len(lead_list))
    print(lead_list[0])
    print(lead_list[0][1].shape)
    random.seed(123)
    random.shuffle(lead_list)
    write = open(file_name+'_onebeat.pkl', 'wb')
    pickle.dump(lead_list, write)
    write.close() 



split_beat("train_all_shuffle",1,True)
split_beat("test_all_shuffle",1,False)








heartbeat_cnt(lead[4][0])