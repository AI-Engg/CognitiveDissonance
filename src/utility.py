import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

interviewer_ques_list = ['Q1: Are you comfortable having your responses recorded on video and by text?',
                  'Q2: Do you consent to having your video analyzed by facial recognition programs, and snapshots to be used in publication if relevant?',
                  'Q3: In which country did you grow up?',
                  'Q4: What area did you grow up in? A city? Countryside?',
                  'Q5: Are you familiar with the ideas of the conservative vs. liberal sides of the political spectrum?',
                  'Q6: Could you briefly characterize what you feel each party represents?',
                  'Q7: On a scale of 0-9, with 0 being entirely liberal and 9 being entirely conservative, and 4-5 being relatively centre, where would you place your own political standing at the moment?',
                  'Q8: On a scale of 0-4, how strongly do you feel about politics, and your political party in particular? With 0 being completely apolitical, and 4 meaning you feel strongly about and align yourself with the ideals of your party.',
                  'Q9: Did you vote in the last election you were eligible for?',
                  'Q9.5: Do you plan on voting any upcoming elections?',
                  'Q10: Do you have anyone in your life that you care for whose political views oppose your own? Someone conservative? (if liberal) Someone liberal? (if conservative)',
                  'Q10.5: Are they conservative or liberal? ',
                  'Q11a: Do you think you can put yourself in their shoes? If you feel comfortable, try to imagine how they see the world - what their priorities are, and how they might look at the people and relationships around them. Imagine - inside their head, those values are the ones that make the most, irrefutable sense. Do you feel you could put yourself in their frame of mind? Are you comfortable trying?',
                  'Q11b: Do you think you could put yourself in the shoes of someone with _(opposite)_ political values? If you feel comfortable, try to imagine how they see the world - what their priorities are, and how they might look at the people and relationships around them. Imagine - inside their head, those values are the ones that make the most, irrefutable sense. Do you feel you could put yourself in their frame of mind? Are you comfortable trying?',
                    'When you decide whether something is right or wrong, to what extent are the following considerations relevant to your thinking? Please rate each statement using this scale:']
opposing_view = ["Q12: Stay in the mindframe of that other person - that one who so opposes your political values. Try to empathize with their position. We now have a questionnaire about some basic values. Answer the questions as though you held the same values as that person. If you feel that person would answer a question the same way you might, then that's no problem, answer each one how you feel a reasonable, liberal/conservative person would."]
true_view = ['Q13: Thank you for your answers. Now, you can release those opinions, and return to the ones you hold. Moving on, could you tell me a little about your family? What are you parents like? Do you have any siblings? (reverting lens to personal views)Q: We would now like to ask you to answer the moral foundations questionnaire again, now how you actually feel about the questions.']

ques_dict = {'Q1':'Q1: Are you comfortable having your responses recorded on video and by text?',
             'Q2':'Q2: Do you consent to having your video analyzed by facial recognition programs, and snapshots to be used in publication if relevant?',
             'Q3':'Q3: In which country did you grow up?',
             'Q4':'Q4: What area did you grow up in? A city? Countryside?',
             'Q5':'Q5: Are you familiar with the ideas of the conservative vs. liberal sides of the political spectrum?',
             'Q6':'Q6: Could you briefly characterize what you feel each party represents?',
             'Q7':'Q7: On a scale of 0-9, with 0 being entirely liberal and 9 being entirely conservative, and 4-5 being relatively centre, where would you place your own political standing at the moment?',
             'Q8':'Q8: On a scale of 0-4, how strongly do you feel about politics, and your political party in particular? With 0 being completely apolitical, and 4 meaning you feel strongly about and align yourself with the ideals of your party.',
              'Q9':'Q9: Did you vote in the last election you were eligible for?',
              'Q9.5':'Q9.5: Do you plan on voting any upcoming elections?',
              'Q10':'Q10: Do you have anyone in your life that you care for whose political views oppose your own? Someone conservative? (if liberal) Someone liberal? (if conservative)',
              'Q10.5':'Q10.5: Are they conservative or liberal? ',
              'Q11a':'Q11a: Do you think you can put yourself in their shoes? If you feel comfortable, try to imagine how they see the world - what their priorities are, and how they might look at the people and relationships around them. Imagine - inside their head, those values are the ones that make the most, irrefutable sense. Do you feel you could put yourself in their frame of mind? Are you comfortable trying?',
               'Q11b':'Q11b: Do you think you could put yourself in the shoes of someone with _(opposite)_ political values? If you feel comfortable, try to imagine how they see the world - what their priorities are, and how they might look at the people and relationships around them. Imagine - inside their head, those values are the ones that make the most, irrefutable sense. Do you feel you could put yourself in their frame of mind? Are you comfortable trying?',
              'Q12':"Q12: Stay in the mindframe of that other person - that one who so opposes your political values. Try to empathize with their position. We now have a questionnaire about some basic values. Answer the questions as though you held the same values as that person. If you feel that person would answer a question the same way you might, then that's no problem, answer each one how you feel a reasonable, liberal/conservative person would.",
             'Q13':'Q13: Thank you for your answers. Now, you can release those opinions, and return to the ones you hold. Moving on, could you tell me a little about your family? What are you parents like? Do you have any siblings? (reverting lens to personal views)Q: We would now like to ask you to answer the moral foundations questionnaire again, now how you actually feel about the questions.'}




interviewer_ques_list = [text.strip().lower() for text in interviewer_ques_list]
interviewer_ques_list = [re.sub(r'[^\w\d\s]+', '', text) for text in interviewer_ques_list]
opposing_view = [text.strip().lower() for text in opposing_view]
opposing_view = [re.sub(r'[^\w\d\s]+', '', text) for text in opposing_view]
true_view = [text.strip().lower() for text in true_view]
true_view = [re.sub(r'[^\w\d\s]+', '', text)  for text in true_view]

def pre_process(df):
    df.rename(columns = {'Questions':'Question'}, inplace=True)
    for row in df.itertuples():
        key = row.Question[:4]
        key = key.split(':',1)[0]
        key = key.strip()
        #print(key, ques_dict[key])
        if key in ques_dict.keys():
            df.at[row.Index,'Question'] = ques_dict[key]
            
    df['Question'] = df['Question'].str.strip().str.lower().str.replace('[^\w\d\s]', '')
    return df

def add_labels(df):
    df['label'] = 'Null'
    oppo = 0
    count = 0
    #print(interviewer_ques_list)
    for row in df.itertuples():
        #print(row.Question.strip().lower() in interviewer_ques_list)
        #print(row.Question)
        if row.Question in interviewer_ques_list:
            df.at[row.Index,'label'] =  'Neutral' #-1
            count += 1
            #row.label = 0
        elif row.Question in opposing_view:
            oppo = 1
            count += 1
            df.at[row.Index,'label'] = 'Neutral'#-1
        elif oppo == 1 and row.Question not in true_view:
            count += 1
            df.at[row.Index,'label'] =  'Dissonance'#0  #disso
        elif row.Question in true_view:
            oppo = 2
            count += 1
            df.at[row.Index,'label'] = 'Neutral' #-1
        elif oppo == 2:
            count += 1
            df.at[row.Index,'label'] = 'True' # 1  #True
        else:
            print('no idea', row.Index)
            
    return df

def plot_corrs(df, fname,  feature_cols = ['Neutrality', 'Happy','Surprise','Fear', 'Disgust', 'Anger', 'Sadness', 'label']):
    plt.rcParams.update({'font.size': 22})
    #plt.rcParams["figure.figsize"] = [10,25]
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    data_df = df.loc[:, feature_cols]
    result = pd.DataFrame()
    result['label'] = data_df.label
    data_df.plot(kind='kde', subplots=True, layout=(1,7), sharex=False, sharey=False, figsize=(25,8),
                title='Density Plots', fontsize=14)
    plt.savefig(fname + '_density_plots.png', dpi=300)
    
def plot_bars(df, fname, feature_cols = ['Neutrality', 'Happy','Surprise','Fear', 'Disgust', 'Anger', 'Sadness', 'label']):
    plt.rcParams.update({'font.size': 22})
    #plt.rcParams["figure.figsize"] = [10,25]
    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    data_df = df.loc[:, feature_cols]
    data_df.groupby("label").max().plot(kind='bar',  subplots=True, layout=(1,7), sharex=False, title = "Max values",figsize=(25,8), fontsize=14)
    plt.savefig(fname + 'bar_plot_Max_values.png', dpi=300)
    plt.figure()
    data_df.groupby("label").mean().plot(kind='bar',  subplots=True, layout=(1,7), sharex=False, title = "Mean values",  figsize=(25,8), fontsize=14)
    plt.savefig(fname + 'bar_plot_Mean_values.png', dpi=300)
    