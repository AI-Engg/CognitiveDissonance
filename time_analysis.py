import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn import tree
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from utility import pre_process, add_labels, plot_corrs, plot_bars
from tqdm import tqdm 
np.random.seed(123)

data_directory = '../data/'
result_directory = '../Results/time/'
fname = ['output1_W2_2019-11-19_09-59-25.csv', 'output2_WIN_20200123_12_48_24_Pro.csv', 'output3_WIN_20200129_14_38_54_Pro.csv',
         'output4_WIN_20200129_14_56_13_Pro.csv', 'output5_WIN_20200129_15_10_31_Pro.csv', 'output6_WIN_20200301_20_18_27_Pro.csv',
         'output7_WIN_20200320_18_43_50_Pro.csv']
feature_cols = ['Question','Time','Time_in_seconds','Neutrality', 'Happy','Surprise','Fear', 'Disgust', 'Anger', 'Sadness', 'label']


for file in tqdm(fname):  ## Each file
    #print('processing file ', file)
    df = pd.read_csv(data_directory+file)
    df['TimeStamp']= pd.to_datetime(df['TimeStamp']) 
    df['Date'] = df['TimeStamp'].dt.date
    df['Time'] = df['TimeStamp'].dt.time
    #df['Time_in_Sec'] = df.TimeStamp.astype(int)
    df['Hour'] = df['TimeStamp'].dt.hour
    df['Minute'] = df['TimeStamp'].dt.minute
    df['Second'] = df['TimeStamp'].dt.second
    df['Time_in_seconds'] = ((df['TimeStamp'].dt.hour*60+df['TimeStamp'].dt.minute)*60 + df['TimeStamp'].dt.second ).astype(int) #
    df = pre_process(df)  # preprocess data
    df = add_labels(df)   # add labels
    
    df_new = df[df.label != 'Neutral']
    df_new = df_new.loc[:, feature_cols]
    df_TT = df_new.set_index('Time_in_seconds')
    for feat in feature_cols[3:-1]:
        #ax = plt.figure(figsize=(15,15))
        df_TT.groupby('label')[feat].plot(title = feat, legend='True', style='.', fontsize = 15, figsize=(15,5))
        #ax.set_ylabel(feat, fontsize=20)
        plt.savefig(result_directory+file+'_'+feat+'_time.png')
        plt.close()
        
    ques = df_new.Question.values
    ques = list(set([q.strip() for q in ques ]))
    
    for N,q in enumerate(ques):
        df_plot_ques = df_new.loc[df_new['Question'] == q]
        Index_dissonance = df_plot_ques[df_plot_ques["label"]=="Dissonance"].index.tolist() 
        Index_true = df_plot_ques[df_plot_ques['label']=="True"].index.tolist() 
        fig, ax = plt.subplots(1,2, figsize=(15,8))
        fig.suptitle(q, fontsize=16)
        for i,emot in enumerate(['Neutrality', 'Happy','Surprise','Fear', 'Disgust', 'Anger', 'Sadness']):
            ax[0].plot(df_plot_ques.loc[Index_dissonance, emot],marker='o', linestyle='-', linewidth=0.5, label=emot)
            ax[1].plot(df_plot_ques.loc[Index_true, emot], marker='o', markersize=8, linestyle='-', label=emot)
        ax[0].set_title("Dissonance")
        ax[1].set_title("True")
        ax[0].set_ylabel('Value')
        ax[0].legend();
        ax[1].legend();
        fig.savefig(result_directory+file+'_Q_'+str(N)+'_ques.png')
        plt.close()
       
    
    
    