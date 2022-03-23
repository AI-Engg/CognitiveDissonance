# Can Artificial Intelligence Identify Cognitive Dissonance?
Ajit Jaokar<sup>1</sup>, Rachel Sava<sup>2</sup>, Amita Kapoor<sup>3</sup>, Claudio Feijoo<sup>1</sup>
<br>
<sup>1</sup>UPM, <sup>2</sup>University College London, United Kingdom, <sup>3</sup>SRCASW, University of Delhi, India



## Introduction
The 1957 experiment by Festinger introduced the concept of Cognitive dissonance (CD), a state of internal conflict and its influence on a person's perception and behavior.  The theory of Cognitive dissonance is considered one of the most influential psychological experiments of the 21st century. In this paper, we ask the question: Can artificial intelligence identify and model cognitive dissonance? We consider a broad definition of artificial intelligence, including machine learning, deep learning, and AGI (Artificial General Intelligence) techniques. Cognitive dissonance is a complex phenomenon for AI to understand because discomfort arising from the state of dissonance motivates the individual undertake some action to reduce the dissonance by altering their cognition. We use several stages and techniques to model both the detection of dissonance state and the motivational aspects of dissonance. 
Firstly, the advent of modern facial recognition algorithms and AI techniques, for the first time, allows the dissonant state itself to be classified using machine learning classification techniques. Our results show that it is possible to determine and quantify the underlying pattern of distribution between micro-expressions and the subject's mental state, enabling us to identify a dissonant state. Next, we leverage several advanced deep learning techniques to model a dissonant state further. We use generative adversarial networks (GANs) to detect the underlying distributions for dissonant states. Finally, we model the interaction as a reinforcement learning problem to identify action taken to reduce dissonance. 
The motivational aspect of cognitive dissonance is further explored through a cognitive architecture called CLARION. We explore the propensity to take action as a result of dissonance from the framework of ‘fast and slow thinking’ (Kahneman). We propose how these ideas can be incorporated into a Cognitive Architecture, specifically the CLARION model. Our work points to a potential revival of some of the ideas from the early (symbolic) nature of AI through hybrid cognitive architectures like CLARION. Finally, we contrast our work with the current thinking in the evolution of deep learning (non-symbolic approaches) to cope with the complex cognitive phenomenon. 
The paper makes several contributions: We demonstrate AI techniques can be used to detect a complex cognitive phenomenon like cognitive dissonance; We make a case for revisiting hybrid cognitive architectures, and we contextualize our work in the background of other techniques being discussed currently.

## Data Collection

11 subjects (5 female, 6 males, aged 18-24) were selected from a pool of university students in London. Prior to testing, permissions were obtained, and their self-reported political leanings were recorded. Participants answered a series of preparatory conversational questions to increase their confidence speaking directly into the video camera. 

The data is in the **data** folder

## Dependencies
* Scikit Learn
* Pandas
* Numpy
* Matplotlib
* tqdm
* Seaborn
* re
* TensorFlow >=2.5

## Code Organization

* **main.py** This file contains the code for all the classifiers. The result is stored in Results folder. It reads the csv files from the data folder and performs the classification task.

* **time_analysis.py** This python script analysis the data as time series. To get better understanding of the data.

* **utility.py** The script contains helpful functions used by the main code.

* **GAN_RL_agent.ipynb** The Jupyter Notebook contains the code for training a GAN which can generate emotion data with same distribution as obtained from the emotion research lab. This GAN is then used to train a RL agent to detect if the emotions represent a cognitive dissonance state or consonance state.


## Citation


 



