# log-mel spectrum Deep Learning TEST
 
A program that extracts features with log-mel spectrum and finds the accuracy with deep learning.

The definition of log-mel spectrum is from the next paper.<br/>
* Deep Learning for Audio Signal Processing: https://arxiv.org/abs/1905.00078
 
Commentary article[JP]:https://qiita.com/Oka_D/items/9afd11e501a1c540bc1b 

# DEMO
 
"""Machine Learning TEST"""<br>
The accuracy and the loss was about `92.6%` and `0.234` respectively using k-fold cross-validation when the dataset of audio sample was as follows and output to Deep Learning of the machine learning algorithm.

* Name:  Jakobovski / Free Spoken Digit Dataset (FSDD)
* LICENCE: Creative Commons Attribution-ShareAlike 4.0 International
* Link: https://github.com/Jakobovski/free-spoken-digit-dataset

The above result can be executed in `MachineLearning.py`. Download the audio sample from the link above.

# Features
 
This program is a test program to prove my next program.
Link: https://github.com/OkamotoDaiki/log-mel_spectrum

 
# Requirement

* Python 3.8.10
* numpy 1.21.4
* Scikit-learn 1.0.1
* Tensorflow 2.8.0
 
# Installation
 
Place the audio file written in DEMO in the folloing file path and execute `MachineLearning.py`
```
"recordings/*.wav"
```

# Author
* Oka.D.
* okamotoschool2018@gmail.com
