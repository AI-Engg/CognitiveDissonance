# Can Artificial Intelligence Identify Cognitive Dissonance?
Ajit Jaokar<sup>1</sup>, Rachel Sava<sup>2</sup>, Amita Kapoor<sup>3</sup>, Claudio Feijoo<sup>1</sup>
<br>
<sup>1</sup>UPM, <sup>2</sup>University College London, United Kingdom, <sup>3</sup>SRCASW, University of Delhi, India



## Introduction
The 1957 experiment by Festinger introduced cognitive dissonance (CD), a state of internal conflict and its influence on a person's perception and behavior. Cognitive dissonance is a complex phenomenon for AI (machine learning and deep learning) to understand because the discomfort arising from a state of dissonance motivates the individual to undertake some action to reduce the uneasiness and regain cognitive coherence. Following Kahnemann, cognitive dissonance could be modeled as System 1 followed by System 2 because the discomfort felt is implicit, but the actions taken to mitigate the discomfort are explicit. The detection of cognitive dissonance can be seen in two parts: firstly, the detection of a dissonant state, followed by an understanding of the motivational aspects to reduce the state of dissonance. Our results show that it is possible to determine and quantify the underlying distribution pattern between micro-expressions and the subject's mental state, enabling us to identify the dissonant state as a machine learning (classification) problem. Next, we explore the propensity to take action due to dissonance. In this phase, we highlight the limitations of existing machine learning and deep learning strategies and explore the motivational aspect of cognitive dissonance through cognitive architectures. The primary contribution of this paper is that multiple techniques may be needed to detect and model a phenomenon like cognitive dissonance. This work could help lay the foundations for exploring similar complex cognitive phenomena that need understanding similar dynamic and motivational aspects for behavior change.

### Keywords: 
Cognitive dissonance, Artificial Intelligence, Facial emotion detection, HCI/Human-computer interface, Machine learning, Affective computing, Cognitive architectures.


## Data Collection

11 subjects (5 female, 6 males, aged 18-24) were selected from a pool of university students in London. Prior to testing, permissions were obtained, and their self-reported political leanings were recorded. Participants answered a series of preparatory conversational questions to increase their confidence speaking directly into the video camera. 

### Data cleaning pre-processing

The data from the Emotion Research Lab API (ERLAPI) consisted of transcribed text with time stamps and a spreadsheet with seven identified emotions of the subject under investigation for the entire interview, each separated by a time gap of 33 ms. We first deduced the labels from the timestamps in the transcribed interview. The frames lying in the timestamp where the subjects answer the questions are treated as True State. The timestamp range where the subjects answer the questions as the opposing person's view is treated as Dissonance State. Finally, the remaining data was labeled as Neutral State. For this paper, we treated the problem as a problem of binary classification and thus retained the data points corresponding to either True State or Dissonance State for further steps.


## Code Structure

The repo has two folders. 
 * data - This contains the data files
 * src -This folder contains the code files. There are three files:

    * **main.py** This file contains the code for all the classifiers. The result is stored in Results folder. It reads the csv files from the data folder and performs the classification task.

    * **time_analysis.py** This python script analysis the data as time series. To get better understanding of the data.

    * **utility.py** The script contains helpful functions used by the main code.


## Dependencies
* matplotlib==3.5.2
* numpy==1.21.6
* pandas==1.3.5
* scikit_learn==1.1.1
* seaborn==0.11.2
* tqdm==4.64.0
* python 3.7.9

## How to run

1. First clone the repo using  `git clone https://github.com/AI-Engg/CognitiveDissonance.git`
2. Create a new virtual environment using `virtualven cd` 
3. Activate the new environment `source cd/bin/activate`
4. Change into the folder where the repo is `cd CognitiveDissonance`
4. Install the required dependencies using `pip install -r requirements.txt`
5. To get the graphs of Figure 1 run `python src/time_analysis.py`
6. To get results of Figure 2 and Figure 3 run `python src/main.py`

The **Results** directory contains the figures.


## Citation

To cite our work:

 



