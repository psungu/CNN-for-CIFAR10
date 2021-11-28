# CNN-for-CIFAR10

To Run the Next Word Prediction Project, please follow the instructions.

Python 3.8.7 

Virtual enviroment is suggested.


install the packages provided in requirements

-Requirements

import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
from sklearn.utils import shuffle
import tensorflow as tf2
from scipy import ndimage, misc
from sklearn.manifold import TSNE

We will install the last version of the tensorflow 2, but sometimes we need to go back to the tensorflow which cases 
are handled with "from tensorflow.compat import v1 as tf"

-main.py

In main file, we have training and model saver. Main imports the model and reading data functions from model.py

To start the training please be sure that dataset should be in the same path with the all py files. And data folder should be
named as "cifar10_data". 

To run the main file please use the following line from the terminal which directs to the project file.

python main.py

During the training events files and checkpoint file are extracted by tensorflow. In checkpoints, we see saved models during
training. There will be more than one model, for the saved models you will see the following log on your terminal screen:

"This epoch has higher validation accuracy. Save session."

Also some .npy files will be extracted which include flatten tensor to use for tsne plot.

-eval.py

In eval file, we have test function. Eval imports the model to start the session, reading data functions from model.py, and
read the last model which is saved to checkpoint.

to call the eval file from the terminal, one may use the following line:

python eval.py

It will print the test accuracy on the terminal screen

-model.py

There is nothing to do with the model file for training and evaluation. It includes read data, augment training data, learning decay 
and model functions.
