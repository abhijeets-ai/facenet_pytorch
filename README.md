This is a README File for a facenet implementation in pytorch
It contains 3 modules and 2 directories:
    1. main.py
    2. model.py
    3. dataset.py
    4. config.py
    5. models directory
    6. logs directory
    7. README.md

1. main.py:
    This module contains code for running training and testing loops.
    implements early stopping and logging for training & monitoring training cycle
2. model.py
    This module contains Facenet class which implements facenet architecture described in paper:
        https://arxiv.org/pdf/1503.03832.pdf
3. dataset.py
    This file contains TripletDataset class used for loading dataset and feeding data to the model.
    It also implements sampling of hard positives and hard negatives
4. config.py
    This file contains hyperparameters as configs to run different experiments
5. model directory
    stores the checkpoints of the model
6. logs directory
    stores the tensorboard logging event files, each subdirectory contains seperate experiment
    to see logged info run `tensorboard --logdir=logs`
7.  README.md
    This readme file
