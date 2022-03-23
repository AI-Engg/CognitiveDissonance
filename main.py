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
result_directory = '../Results/'
fname = ['output1_W2_2019-11-19_09-59-25.csv', 'output2_WIN_20200123_12_48_24_Pro.csv', 'output3_WIN_20200129_14_38_54_Pro.csv',
         'output4_WIN_20200129_14_56_13_Pro.csv', 'output5_WIN_20200129_15_10_31_Pro.csv', 'output6_WIN_20200301_20_18_27_Pro.csv',
         'output7_WIN_20200320_18_43_50_Pro.csv']
feat_cols = ['Neutrality', 'Happy','Surprise','Fear', 'Disgust', 'Anger', 'Sadness', 'label']


for file in tqdm(fname):  ## Each file
    #print('processing file ', file)
    df = pd.read_csv(data_directory+file)
    corr = df.corr()

    # plot the heatmap
    plt.figure(figsize = (25,25))
    hm = sns.heatmap(corr, annot=True, fmt=".1",
            xticklabels=corr.columns,
            yticklabels=corr.columns)
    figure = hm.get_figure()    
    figure.savefig(result_directory+file+'heatmap_all_features.png', dpi=400)
    df = pre_process(df)  # preprocess data
    df = add_labels(df)   # add labels
    plot_corrs(df,fname = result_directory+file+'_corrs', feature_cols = feat_cols)
    plot_bars(df, fname = result_directory+file+'_corrs', feature_cols = feat_cols)


    X = df.loc[:, feat_cols[:-1]]
    y = df.label

    le = preprocessing.LabelEncoder()
    le.fit(list(set(df.label.values)))

    y = le.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
    
    
    # calculate the correlation matrix
    corr = X.corr()

    # plot the heatmap
    plt.figure(figsize = (25,25))
    hm = sns.heatmap(corr, annot=True, fmt=".1",
        xticklabels=corr.columns,
        yticklabels=corr.columns)
    figure = hm.get_figure()    
    figure.savefig(result_directory+file+'heatmap_selected_features.png', dpi=400)
    
    ## Apply classifiers
    dict_classifiers = {
    "Logreg": LogisticRegression(solver='lbfgs'),
    #"NN": KNeighborsClassifier(),
    "LinearSVM": SVC(probability=True, kernel='linear'), #class_weight='balanced'
    "RBF_SVM": SVC(probability=True, kernel='rbf'),
    #"GBC": GradientBoostingClassifier(),
    "DT": tree.DecisionTreeClassifier(),
    "RF": RandomForestClassifier(),
    #"NB": GaussianNB(),
    }
    results = pd.DataFrame()
    results['Metrics'] = ['Accuracy_Train', 'Precision_Train', 'Recall_Train', 'Accuracy_Test', 
                              'Precision_Test','Recall_Test']

    ft = pd.DataFrame([[i] for i in feat_cols[:-1]], columns =['Features']) 
    results.set_index(['Metrics'], inplace=True)
    
    for model, model_instantiation in dict_classifiers.items():  ## for different classifiers
        y_score = model_instantiation.fit(X_train, y_train)
        y_pred = pd.DataFrame( model_instantiation.predict(X_train)).reset_index(drop=True)
        Recall_Train,Precision_Train, Accuracy_Train  = recall_score(y_train, y_pred, average='micro'), precision_score(y_train, y_pred, average='micro'), accuracy_score(y_train, y_pred)
        y_pred = pd.DataFrame( model_instantiation.predict(X_test)).reset_index(drop=True)
        Recall_Test = recall_score(y_test, y_pred,average='micro')
        Precision_Test = precision_score(y_test, y_pred, average='micro')
        Accuracy_Test = accuracy_score(y_test, y_pred)
    
        if model in ['DT', 'RF']:
            feat_imp = dict(zip(X.columns, dict_classifiers[model].feature_importances_))
            ft['Model_'+model] =  ft['Features'].map(feat_imp) 

        results['Model_'+model] = [Accuracy_Train, Precision_Train, Recall_Train, Accuracy_Test, 
                            Precision_Test, Recall_Test]   
    results.to_csv(result_directory+file + '_metrics.csv') 
    ft.to_csv(result_directory+file + '_feature_importance.csv')
    for model, model_instantiation in dict_classifiers.items():
        predictions = model_instantiation.predict(X_test)
        plt.figure(figsize = (10,10))
        ax = plt.axes()
        hm = sns.heatmap(confusion_matrix(y_test, predictions), annot=True, fmt="d", ax = ax,
            xticklabels= list(set(df.label.values)),
            yticklabels=list(set(df.label.values)))
        ax.set_title('Model ' + model)
        figure = hm.get_figure()    
        figure.savefig(result_directory+file+'_'+ model+'_confusion_matrix.png', dpi=400)