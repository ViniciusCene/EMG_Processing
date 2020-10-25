# IMPORT LIBRARIES AND DATA
#%reset
%matplotlib qt 

import numpy as np
import pandas as pd
import scipy.io as sio
from scipy import stats
from numpy import savetxt
from scipy.io import loadmat
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.neural_network import MLPClassifier


def load_nina_data(folder, database, subj, exerc):
    'load_nina_data(folder, database, subj, exerc):'
    
    # folder with the data
    path = (folder + '\\' + database + '\\RAW\\S' + str(subj) + '_E' + str(exerc)  + '_A1.mat')
    allData = sio.loadmat(path)
    emgRAWData = np.double(allData['emg'])
    labels = np.double(allData['restimulus'])
    return emgRAWData, labels


def load_iee_data(folder, subject, exerc, session):
    """load_iee_data(folder, subject, exerc, session)."""
    #path = (folder + '\\' + database + '\\RAW\\S' + str(subj) + '_E' + str(exerc)  + '_A1.mat')
    file = (folder + '\\S' + subject + '_' + exerc + session + '.mat')
   
    allData = sio.loadmat(file)
    emgData = np.double(allData['emg'])
    
    labels = allData['restimulus']
    labels = np.double(labels)
    return emgData, labels
	
	
def removeNullData(emgData):

    numCH = np.size(emgData,1)
    numSamples = np.size(emgData)
    col = numCH-1

    while col >= 0:
        occurrences = np.where(emgData[:,col] == 0)
        if np.size(occurrences) == numSamples:
            np.delete(emgData[:,col], 0, col)
        else:
            emgData[occurrences,col] = 10E-6
        col = col-1
        
    print("NULL VALUES RESOLVED")    


#AVT FILTER
def avt_filter(emgData):
    """Statistical Filtering."""
    emg = np.copy(emgData)
    ff1 = 0.8
    ff2 = 0.2
    sF = 2000
    tS = len(emg[:, 0])
    twS = int(0.4*sF) - 1
    overlapS = int(0.01*sF)
    nCh = np.shape(emg)[1]

    for i in range(0, tS, overlapS):
        if (i + twS) > tS:
            return emg
        for ch in range(nCh):

            mean = (np.mean(emg[i:i+twS-20, ch])*ff1 + np.mean(emg[i+twS-20:i+twS, ch])*ff2)
            std = (np.std(emg[i:i+twS-20, ch])*ff1 + np.std(emg[i+twS-20:i+twS, ch])*ff2)
            highLim = mean + std
            lowLim = mean - std
            filteredData = np.where(emg[i:i+twS, ch] < lowLim, mean, emg[i:i+twS, ch])
            filteredData = np.where(filteredData > highLim, mean, filteredData)
            emg[i:i+twS, ch] = filteredData

    return emg
	
	
# FEATS
def feature_extraction(signal):
    """Statistical Filtering."""
    
    sF = 2000
    tS = len(signal[:, 0])
    overlapS = int(0.01*sF)
    twS = int(0.4*sF) - 1
    nCh = np.shape(signal)[1] 
    idx = 0
    lengthFeat = int(np.round(((tS-twS)/overlapS)))
    
    label = np.zeros((lengthFeat, 1))
    feat_rms = np.zeros((lengthFeat, 12))
    feat_std = np.zeros((lengthFeat, 12))
    feat_var = np.zeros((lengthFeat, 12))
    feat_mav = np.zeros((lengthFeat, 12))
    feat_DES = np.zeros((lengthFeat, 12))
    
    idx = 0    
    for i in range(0, tS-twS, overlapS):
        for ch in range(nCh):
            if ch==0:
                label[idx,0] = stats.mode(signal[i:i+twS, 0])[0]
            else:    
                feat_rms[idx,ch-1] = np.sqrt(np.mean(np.square(signal[i:i+twS, ch])))
                feat_std[idx,ch-1] = np.std(signal[i:i+twS, ch])
                feat_var[idx,ch-1] = np.var(signal[i:i+twS, ch])
                feat_mav[idx,ch-1] = np.mean(signal[i:i+twS, ch]) 
        if idx==0:    
            a = feat_mav[idx,:] #np.mean(signal[i:i+twS, 1:nCh])
            b = np.ones((nCh-1))
        else: 
            b = a
            a = feat_mav[idx,:]
        c = np.concatenate(([a], [b]), axis=0)   
        pca = PCA(n_components=1)
        pca.fit(c)
        #print(pca.components_)
        feat_DES[idx,:] = pca.components_
        idx = idx+1    
    
    filtered_DES = moving_average(feat_DES)
    feats = np.concatenate((label, feat_rms, feat_std, feat_var, feat_mav, filtered_DES), axis=1)
    
    return feats


def moving_average(data):
    filtered_DES = np.zeros(np.shape(data))
    data = np.absolute(data)

    N=10
    for i in range(12):
        print(i)
        data_padded = np.pad(data[:,i], (N//2, N-1-N//2), mode='edge')
        filtered_DES[:,i] = np.convolve(data_padded, np.ones((N,))/N, mode='valid') 
        
    return filtered_DES     	
	
	
# Normalize Features
def normalize_feats(data):
    
    nFeats = np.shape(data)[1]
    normFeats = np.zeros(np.shape(data))
    normFeats[:, 0] = data[:, 0]
    i=1                                                     #data[:,0] is for the label, so we are starting from data[:,1]
    while i < nFeats:
        normFeats[:,i] = data[:,i] / np.max(data[:,i])  
        i=i+1
        
    return normFeats


def movement_split(labels, numRep, numMov):
    
    labels = normFeats[:,0]
    movACT = np.zeros((numMov+1,numRep))
    movDEACT = np.zeros((numMov+1,numRep))
    train = np.zeros([1, 61])    
    teste = np.zeros([1, 61])    
    i = 1
    rep = 0
    
    while i < len(labels):
    
        if labels[i - 1] == 0 and labels[i] != 0:
            movACT[int(labels[i]-1), rep] = i - 1
            #rep = rep + 1
        
        elif labels[i - 1] != 0 and labels[i] == 0:
            movDEACT[int(labels[i-1]-1), rep] = i 
            rep = rep + 1
            
        if rep == numRep:
            rep = 0
        i = i + 1
    
    movACT = np.delete(movACT, -1, 0)  #Parse matrix
    movDEACT = np.delete(movDEACT, -1, 0)
    
    for i in range(numMov):
        if i == 0:
            train = np.concatenate((train, normFeats[0:int(down[i,3]+1),:]), axis=0)
        else:
            train = np.concatenate((train, normFeats[int(down[i-1,5]+1):int(down[i,3]+1),:]), axis=0)    
        teste = np.concatenate((teste, normFeats[int(down[i,3]+1):int(down[i,5]+1),:]), axis=0)
    
    return train, teste
    

# PRE-PROCESSING MAIN FUNCTION

# Declare path, database and exercise type
folder = 'C:\\Users\\vinic\\Desktop\\EMG'
database = 'NAMP\\'
subj = '1'
exerc = '1'

# Load raw data and labels
[rawEMG, label] = load_nina_data(folder, database, subj, exerc)

# Identify the number of repetitions performed to each movement
if database == 'NAMP\\': 
    numRep = 6
elif database == 'AMP\\':
    if subj != '1': 
        numRep = 6    
    else:    
        numRep = 6    
#elif database == 'IEE\\':       
# FAZER O PARSING PARA A BASE IEE DEPOIS
else:
    print('INVALID DATABASE!')

# Retify and filter the data    
absEMG = np.absolute(rawEMG)
emgAVT = avt_filter(absEMG)

# Assure that each assay will end with 0
if label[-1] != 0:
    label[-1] = 0
    
# Concatanation of label and data    
signal = np.concatenate((label, emgAVT), axis=1)    

# Feature extraction and normalization
feats = feature_extraction(signal)    
normFeats = normalize_feats(feats)

# Calculate number of different classes (17 + rest for the three databases)
numClasses = len(np.unique(normFeats[:,0]))-1    

# Generate train and test data to be classified
[train, teste] = movement_split(normFeats, numRep, numClasses)

# Save CSV files for train and test 
savetxt('train.csv', train, delimiter=',')
savetxt('teste.csv', teste, delimiter=',')	